import whisper
import os
import argparse
from pathlib import Path
import json

def transcribe_video(video_name, model_size="base"):
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"

    video_path = data_dir / video_name
    output_path = data_dir / f"{video_name}.json"

    model = whisper.load_model(model_size)
    print(f"Transcribing {video_path} with Whisper-{model_size}...")
    result = model.transcribe(str(video_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save only the segments (timestamped transcript) to the output file
    segments = result.get("segments", [])
    simplified_result = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simplified_result, f, indent=2)

    print(f"Simplified timestamped transcript saved to {output_path}")
    return simplified_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Filename of the local video in the data folder (e.g., sample.mp4)")
    parser.add_argument("--model", default="base", help="Whisper model size: tiny, base, small, medium, large")
    args = parser.parse_args()

    transcribe_video(args.video, args.model)