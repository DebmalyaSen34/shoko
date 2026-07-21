import os
from unittest import mock

from src.generator.media import (
    _adaptive_frame_count,
    _extract_video_frames,
    _frame_offsets_for_duration,
)


def test_adaptive_frame_count_scales_by_duration():
    assert _adaptive_frame_count(0) == 1
    assert _adaptive_frame_count(2.0) == 3
    assert _adaptive_frame_count(2.1) == 5
    assert _adaptive_frame_count(5.1) == 7
    assert _adaptive_frame_count(10.1) == 9


def test_frame_offsets_include_safe_start_and_end():
    offsets = _frame_offsets_for_duration(12.0)

    assert len(offsets) == 9
    assert offsets[0] == 0.05
    assert offsets[-1] == 11.9
    assert offsets == sorted(offsets)


def test_frame_offsets_deduplicate_for_subsecond_clips():
    offsets = _frame_offsets_for_duration(0.001, frame_count=9)

    assert offsets == [0.0, 0.001]


@mock.patch("src.generator.media.subprocess.run")
def test_extract_video_frames_uses_adaptive_full_clip_offsets(mock_run, tmp_path):
    mock_run.return_value = mock.Mock(returncode=0, stderr="")

    frame_paths = _extract_video_frames(
        "clip.mp4",
        str(tmp_path),
        duration_s=6.0,
    )

    assert len(frame_paths) == 7
    assert frame_paths[0] == os.path.join(str(tmp_path), "frame_001.jpg")
    assert frame_paths[-1] == os.path.join(str(tmp_path), "frame_007.jpg")
    offsets = [call.args[0][3] for call in mock_run.call_args_list]
    assert offsets[0] == "0.050"
    assert offsets[-1] == "5.900"
