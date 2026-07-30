use std::{
    env,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use tauri::Manager;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

struct BackendManager {
    base_url: Mutex<Option<String>>,
    child: Mutex<Option<Child>>,
    shutdown_token: String,
}

impl BackendManager {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        Self {
            base_url: Mutex::new(None),
            child: Mutex::new(None),
            shutdown_token: format!("{}-{}", std::process::id(), nonce),
        }
    }

    fn start(&self, app: &tauri::AppHandle) -> Result<String, String> {
        if let Some(base_url) = self.base_url.lock().map_err(lock_error)?.clone() {
            return Ok(base_url);
        }

        let port = find_free_port()?;
        let base_url = format!("http://127.0.0.1:{port}");
        let child = self.spawn_backend(app, port)?;

        {
            let mut child_slot = self.child.lock().map_err(lock_error)?;
            *child_slot = Some(child);
        }

        self.wait_for_health(&base_url)?;

        let mut base_slot = self.base_url.lock().map_err(lock_error)?;
        *base_slot = Some(base_url.clone());
        Ok(base_url)
    }

    fn spawn_backend(&self, app: &tauri::AppHandle, port: u16) -> Result<Child, String> {
        if let Ok(command) = env::var("LOKA_BACKEND_COMMAND") {
            return spawn_shell_command(&command, port, &self.shutdown_token);
        }

        if let Ok(binary) = env::var("LOKA_BACKEND_BINARY") {
            return spawn_backend_binary(PathBuf::from(binary), port, &self.shutdown_token);
        }

        for candidate in backend_binary_candidates(app) {
            if candidate.exists() {
                return spawn_backend_binary(candidate, port, &self.shutdown_token);
            }
        }

        spawn_python_dev_backend(port, &self.shutdown_token)
    }

    fn wait_for_health(&self, base_url: &str) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if let Some(status) = http_request("GET", base_url, "/health", None) {
                if status.starts_with("HTTP/1.1 200") || status.starts_with("HTTP/1.0 200") {
                    return Ok(());
                }
            }

            if let Ok(mut child_slot) = self.child.lock() {
                if let Some(child) = child_slot.as_mut() {
                    if let Ok(Some(status)) = child.try_wait() {
                        return Err(format!("Backend exited before becoming healthy: {status}"));
                    }
                }
            }

            thread::sleep(Duration::from_millis(250));
        }

        Err(format!(
            "Backend did not become healthy at {base_url}/health"
        ))
    }

    fn stop(&self) {
        let base_url = self.base_url.lock().ok().and_then(|slot| slot.clone());
        if let Some(base_url) = base_url {
            let _ = http_request("POST", &base_url, "/shutdown", Some(&self.shutdown_token));
        }

        if let Ok(mut child_slot) = self.child.lock() {
            if let Some(child) = child_slot.as_mut() {
                let deadline = Instant::now() + Duration::from_secs(3);
                while Instant::now() < deadline {
                    if child.try_wait().ok().flatten().is_some() {
                        *child_slot = None;
                        return;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                let _ = child.kill();
                let _ = child.wait();
            }
            *child_slot = None;
        }
    }
}

fn lock_error<T>(error: std::sync::PoisonError<T>) -> String {
    format!("Backend state lock failed: {error}")
}

fn find_free_port() -> Result<u16, String> {
    TcpListener::bind("127.0.0.1:0")
        .map_err(|error| format!("Could not bind local backend port: {error}"))?
        .local_addr()
        .map(|addr| addr.port())
        .map_err(|error| format!("Could not read local backend port: {error}"))
}

fn backend_binary_candidates(app: &tauri::AppHandle) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let binary_name = if cfg!(windows) {
        "backend-server.exe"
    } else {
        "backend-server"
    };

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join(binary_name));
        candidates.push(resource_dir.join("backend").join(binary_name));
    }

    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(exe_dir.join(binary_name));
            candidates.push(exe_dir.join("backend").join(binary_name));
        }
    }

    candidates
}

fn spawn_backend_binary(binary: PathBuf, port: u16, shutdown_token: &str) -> Result<Child, String> {
    let mut command = Command::new(&binary);
    command
        .env("LOKA_BACKEND_PORT", port.to_string())
        .env("PORT", port.to_string())
        .env("LOKA_BACKEND_SHUTDOWN_TOKEN", shutdown_token)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    configure_hidden_console(&mut command);

    command
        .spawn()
        .map_err(|error| format!("Could not start bundled backend {:?}: {error}", binary))
}

fn spawn_python_dev_backend(port: u16, shutdown_token: &str) -> Result<Child, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let backend_dir = manifest_dir
        .parent()
        .map(|path| path.join("backend"))
        .ok_or_else(|| "Could not resolve backend directory".to_string())?;
    let python = env::var("LOKA_BACKEND_PYTHON").unwrap_or_else(|_| {
        let venv_python = if cfg!(windows) {
            backend_dir.join(".venv").join("Scripts").join("python.exe")
        } else {
            backend_dir.join(".venv").join("bin").join("python")
        };

        if venv_python.exists() {
            venv_python.to_string_lossy().to_string()
        } else if cfg!(windows) {
            "python".to_string()
        } else {
            "python3".to_string()
        }
    });

    let mut command = Command::new(&python);
    command
        .current_dir(&backend_dir)
        .arg("-m")
        .arg("uvicorn")
        .arg("server:app")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .env("LOKA_BACKEND_PORT", port.to_string())
        .env("PORT", port.to_string())
        .env("LOKA_BACKEND_SHUTDOWN_TOKEN", shutdown_token)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    configure_hidden_console(&mut command);

    command.spawn().map_err(|error| {
        format!(
            "Could not start development backend with {python} in {:?}: {error}",
            backend_dir
        )
    })
}

fn spawn_shell_command(command: &str, port: u16, shutdown_token: &str) -> Result<Child, String> {
    let mut child_command = if cfg!(windows) {
        let mut command_runner = Command::new("cmd");
        command_runner.arg("/C").arg(command);
        command_runner
    } else {
        let mut command_runner = Command::new("sh");
        command_runner.arg("-c").arg(command);
        command_runner
    };

    child_command
        .env("LOKA_BACKEND_PORT", port.to_string())
        .env("PORT", port.to_string())
        .env("LOKA_BACKEND_SHUTDOWN_TOKEN", shutdown_token)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    configure_hidden_console(&mut child_command);

    child_command
        .spawn()
        .map_err(|error| format!("Could not start backend command `{command}`: {error}"))
}

#[cfg(windows)]
fn configure_hidden_console(command: &mut Command) {
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn configure_hidden_console(_command: &mut Command) {}

fn http_request(
    method: &str,
    base_url: &str,
    path: &str,
    shutdown_token: Option<&str>,
) -> Option<String> {
    let port = base_url.rsplit(':').next()?.parse::<u16>().ok()?;
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let token_header = shutdown_token
        .map(|token| format!("x-loka-shutdown-token: {token}\r\n"))
        .unwrap_or_default();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{token_header}Connection: close\r\nContent-Length: 0\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).ok()?;

    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    Some(response.lines().next().unwrap_or_default().to_string())
}

#[tauri::command]
async fn backend_base_url(
    app: tauri::AppHandle,
    manager: tauri::State<'_, Arc<BackendManager>>,
) -> Result<String, String> {
    let manager = manager.inner().clone();
    tauri::async_runtime::spawn_blocking(move || manager.start(&app))
        .await
        .map_err(|error| format!("Backend startup task failed: {error}"))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let backend_manager = Arc::new(BackendManager::new());
    let shutdown_manager = backend_manager.clone();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(backend_manager)
        .on_window_event(move |_window, event| {
            if matches!(*event, tauri::WindowEvent::Destroyed) {
                shutdown_manager.stop();
            }
        })
        .invoke_handler(tauri::generate_handler![backend_base_url])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
