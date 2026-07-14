import gzip
import os
import xml.etree.ElementTree as ET
from config.premiere_pro_conf import SEQUENCE_CLASS, VIDEO_CLIP_TRACK_ITEM, MEDIA_CLASS, SUBCLIP_CLASS, AUDIO_CLIP_TRACK_ITEM, MASTERCLIP_CLASS
from .utils import win_basename, ticks_to_s, ticks_to_tc

def parse_prproj_file(prproj_path: str, output_xml_path: str) -> str:
    """Parses the specified .prproj file and writes the extracted XML content to the specified output path.

    Args:
        prproj_path (str): Path to the .prproj file to be parsed.
        output_xml_path (str): Path where the extracted XML content will be written.

    Raises:
        FileNotFoundError: If the specified .prproj file does not exist.
        ValueError: If the specified file is not a .prproj file.

    Returns:
        str: The path to the output XML file where the extracted content has been written.
    """

    if not os.path.exists(prproj_path):
        raise FileNotFoundError(f"The specified .prproj file does not exist: {prproj_path}")
    
    if not prproj_path.endswith('.prproj'):
        raise ValueError(f"The specified file is not a .prproj file: {prproj_path}")

    with gzip.open(prproj_path, 'rb') as f:
        xml = f.read()

    with open(output_xml_path, 'wb') as out:
        out.write(xml)
    return output_xml_path


def parse_xml_file(xml_path: str) -> dict:
    """Parses the specified XML file and extracts relevant data into a structured dictionary.

    Args:
        xml_path (str): Path to the XML file to be parsed.

    Raises:
        FileNotFoundError: If the specified XML file does not exist.
        ValueError: If the specified file is not an XML file.

    Returns:
        dict: A dictionary containing the extracted data, including sequence name, total duration, frame size, media assets, video and audio timelines, and master clip names.
    """

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"The specified XML file does not exist: {xml_path}")
    
    if not xml_path.endswith('.xml'):
        raise ValueError(f"The specified file is not an XML file: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Build object lookup maps
    object_map = {}   # ObjectID  -> element
    uref_map   = {}   # ObjectURef -> element
    for elem in root.iter():
        oid  = elem.get("ObjectID")
        uref = elem.get("ObjectURef")
        if oid:  object_map[oid]  = elem
        if uref: uref_map[uref]   = elem

    # Sequence
    seq = next((e for e in root.iter() if e.get("ClassID") == SEQUENCE_CLASS), None)
    seq_name = seq.findtext("Name") if seq is not None else "Unknown"

    # Detect frame size from first video track item
    first_vti = next((e for e in root.iter() if e.get("ClassID") == VIDEO_CLIP_TRACK_ITEM), None)
    frame_size = "unknown"
    if first_vti is not None:
        fr = first_vti.findtext("FrameRect")
        if fr:
            parts = fr.split(",")
            if len(parts) == 4:
                frame_size = f"{parts[2]}x{parts[3]}"

    # Media assets (all imported files)
    media_assets = {}   # FileKey -> dict
    for med in root.iter():
        if med.get("ClassID") != MEDIA_CLASS:
            continue

        def t(tag):
            el = med.find(tag)
            return el.text.strip() if el is not None and el.text else None

        title       = t("Title")
        file_key    = t("FileKey")
        actual_path = t("ActualMediaFilePath")
        alt_path    = t("FilePath")
        history     = t("MediaFileHistory0")

        name = title or win_basename(actual_path) or win_basename(alt_path) or "unknown"
        ext  = os.path.splitext(name)[1].lower()
        #NOTE that we classify media types based on file extension, which may not always be accurate.
        mtype = (
            "audio" if ext in (".wav", ".mp3", ".aac", ".aif", ".aiff", ".m4a", ".flac")
            else "video" if ext in (".mp4", ".mov", ".avi", ".mxf", ".r3d", ".mkv")
            else "other"
        )

        if file_key:
            media_assets[file_key] = {
                "filename":          name,
                "type":              mtype,
                "actual_path":       actual_path,
                "original_mac_path": history,
            }

    # SubClip name map
    # SubClip elements link timeline items to their source files by name
    subclip_names = {}   # ObjectID -> clip name
    for sc in root.iter():
        if sc.get("ClassID") != SUBCLIP_CLASS:
            continue
        name_el = sc.find("Name")
        if name_el is not None and name_el.text:
            subclip_names[sc.get("ObjectID")] = name_el.text.strip()

    # Parse a list of track items 
    def parse_items(items):
        result = []
        for item in items:
            ti = item.find('.//TrackItem[@Version="3"]')
            start_t = ti.findtext("Start") if ti is not None else None
            end_t   = ti.findtext("End")   if ti is not None else None

            sc_ref    = item.find(".//SubClip")
            clip_name = None
            if sc_ref is not None:
                sc_oid = sc_ref.get("ObjectRef")
                if sc_oid:
                    sc_elem = object_map.get(sc_oid)
                    if sc_elem is not None:
                        clip_name = sc_elem.findtext("Name")
                    if not clip_name:
                        clip_name = subclip_names.get(sc_oid)

            result.append({
                "clip":       clip_name or "unknown",
                "start_tc":   ticks_to_tc(start_t),
                "end_tc":     ticks_to_tc(end_t),
                "start_s":    ticks_to_s(start_t),
                "end_s":      ticks_to_s(end_t),
                "duration_s": round(ticks_to_s(end_t) - ticks_to_s(start_t), 3),
            })
        result.sort(key=lambda x: x["start_s"])
        return result

    video_timeline = parse_items(
        [e for e in root.iter() if e.get("ClassID") == VIDEO_CLIP_TRACK_ITEM]
    )
    all_audio = parse_items(
        [e for e in root.iter() if e.get("ClassID") == AUDIO_CLIP_TRACK_ITEM]
    )

    audio_exts = {".wav", ".mp3", ".aac", ".aif", ".aiff", ".m4a", ".flac"}
    dedicated_audio = [c for c in all_audio
                       if os.path.splitext(c["clip"])[1].lower() in audio_exts]
    embedded_audio  = [c for c in all_audio if c not in dedicated_audio]

    # Master clip names (project bin)
    master_clip_names = []
    for mc in root.iter():
        if mc.get("ClassID") != MASTERCLIP_CLASS:
            continue
        name = mc.findtext("Name")
        if name:
            master_clip_names.append(name)

    # Total timeline duration 
    total_s = max((c["end_s"] for c in video_timeline), default=0.0)
    h, rem  = divmod(total_s, 3600)
    m, s    = divmod(rem, 60)
    total_tc = f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

    # Structure the output data
    output_data ={
        "sequence_name":     seq_name,
        "total_duration_s":  total_s,
        "total_duration_tc": total_tc,
        "fps":               25,
        "frame_size":        frame_size,
        "summary": {
            "total_video_clips_on_timeline":  len(video_timeline),
            "total_audio_items_on_timeline":  len(all_audio),
            "unique_media_assets_imported":   len(media_assets),
            "dedicated_audio_track_segments": len(dedicated_audio),
        },
        "media_assets": {
            "video": [v for v in media_assets.values() if v["type"] == "video"],
            "audio": [v for v in media_assets.values() if v["type"] == "audio"],
            "other": [v for v in media_assets.values() if v["type"] == "other"],
        },
        "video_timeline": video_timeline,
        "audio_timeline": {
            "dedicated_audio_tracks": dedicated_audio,
            "embedded_video_audio":   embedded_audio,
        },
        "all_imported_master_clips": sorted(set(master_clip_names)),
    }
    return output_data

def parse_prproj_to_json(prproj_path: str, output_json_path: str):
    """Parses the specified .prproj file and writes the extracted data to a JSON file.

    Args:
        prproj_path (str): Path to the .prproj file to be parsed.
        output_json_path (str): Path where the extracted data will be written as JSON.
    """
    temp_xml_path = output_json_path + ".tmp.xml"
    parse_prproj_file(prproj_path, temp_xml_path)
    data = parse_xml_file(temp_xml_path)
    
    import json
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    os.remove(temp_xml_path)