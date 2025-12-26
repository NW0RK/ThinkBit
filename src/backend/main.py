from __future__ import annotations

import hashlib
import logging
import os
import sys
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from sqlalchemy import and_
from sqlalchemy.orm import Session

# Force UTF-8 encoding for stdout/stderr on Windows to avoid charmap errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from .db import get_db, init_db
# Removed User and auth-related models from import
from .models import CensorSegment, ProcessStatus, ProcessedMedia, utc_now
# Removed auth-related schemas from import
from .schemas import HealthResponse, MediaListResponse, MediaResponse, MessageResponse, RawFileResponse, SegmentResponse, StatsResponse
from .services.pipeline_wrapper import process_media

# Configure logging explicitly to ensure FileHandler is attached even if Uvicorn configures the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Check if FileHandler already exists to avoid duplicates on reload
has_file_handler = any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith("backend.log") for h in logger.handlers)

if not has_file_handler:
    file_handler = logging.FileHandler("backend.log", mode='w')
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    # Also attach to uvicorn loggers to capture server logs
    for log_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uvicorn_logger = logging.getLogger(log_name)
        uvicorn_logger.addHandler(file_handler)

# Force root logger to propagate if it was disabled
logger.propagate = True
logger.info("Logging configured.")

# Ensure we also output to console (Uvicorn usually does this, but good to be safe)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

MAX_FILE_SIZE = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".flac", ".m4a", ".webm"}
ALLOWED_MIMETYPES = {"video/mp4", "video/quicktime", "video/x-matroska", "video/x-msvideo", 
                     "audio/wav", "audio/mpeg", "audio/flac", "audio/x-m4a", "audio/mp4",
                     "video/webm", "audio/webm"}


def _ensure_directories():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _detect_input_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".wav", ".mp3", ".flac", ".m4a"}:
        return "audio"
    if ext in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        return "video"
    return "video"


def _compute_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _to_response(media: ProcessedMedia) -> MediaResponse:
    input_name = Path(media.input_path).name if media.input_path else None
    output_name = Path(media.output_path).name if media.output_path else None
    return MediaResponse(
        id=media.id,
        input_path=input_name,
        output_path=output_name,
        input_type=media.input_type,
        filter_audio=media.filter_audio,
        filter_video=media.filter_video,
        status=media.status,
        progress=media.progress,
        current_activity=media.current_activity,
        logs=media.logs.split("\n") if media.logs else [],
        error_message=media.error_message,
        created_at=media.created_at,
        updated_at=media.updated_at,
        segments=[
            SegmentResponse(
                id=seg.id,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                action_type=seg.action_type,
                reason=seg.reason,
            )
            for seg in media.segments
        ],
    )


def _validate_upload(filename: str, content_type: str | None, size: int) -> None:
    """Validate uploaded file with security checks."""
    # Security: Prevent path traversal and malicious filenames
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Validate extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type not allowed: {ext}")
    
    # Validate MIME type
    if content_type and content_type not in ALLOWED_MIMETYPES:
        raise HTTPException(status_code=400, detail=f"Invalid content type: {content_type}")
    
    # Validate file size
    if size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB")
    
    # Additional security: reject empty files
    if size == 0:
        raise HTTPException(status_code=400, detail="Empty files not allowed")


def range_file_response(
    path: Path, 
    range_header: str | None, 
    media_type: str = "application/octet-stream"
):
    """
    Returns a StreamingResponse if a Range header is present,
    otherwise returns a standard FileResponse.
    """
    file_size = path.stat().st_size
    if not range_header:
        return FileResponse(path=path, filename=path.name, media_type=media_type)

    try:
        # Parse standard Range header: "bytes=start-end"
        unit, ranges = range_header.split("=")
        if unit.strip().lower() != "bytes":
             return FileResponse(path=path, filename=path.name, media_type=media_type)
        
        start_str, end_str = ranges.split("-")
        
        # Handle suffix range (e.g. bytes=-500) -> last 500 bytes
        if not start_str and end_str:
            suffix_length = int(end_str)
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        else:
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
        
        if start >= file_size:
            # Requested range not satisfiable
            return FileResponse(path=path, filename=path.name, media_type=media_type)

        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iter_file():
            with open(path, "rb") as f:
                f.seek(start)
                bytes_read = 0
                while bytes_read < chunk_size:
                    chunk = f.read(min(8192, chunk_size - bytes_read))
                    if not chunk:
                        break
                    yield chunk
                    bytes_read += len(chunk)

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
        }
        
        return StreamingResponse(
            iter_file(),
            status_code=206,
            headers=headers,
            media_type=media_type,
        )

    except Exception as e:
        logger.error(f"Error parsing range header '{range_header}': {e}")
        # Fallback
        return FileResponse(path=path, filename=path.name, media_type=media_type)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_directories()
    init_db()
    logger.info("Backend started")
    yield
    logger.info("Backend stopped")


app = FastAPI(
    title="AegisAI (No Auth)",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/api",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/stats", response_model=StatsResponse)
def get_stats(
    db: Session = Depends(get_db)
):
    """Get statistics for all media."""
    # Removed user_id filtering
    query = db.query(ProcessedMedia)
    total_media = query.count()
    
    media_ids = [m.id for m in query.all()]
    total_segments = db.query(CensorSegment).filter(CensorSegment.media_id.in_(media_ids)).count() if media_ids else 0

    by_status = {
        ProcessStatus.CREATED: query.filter(ProcessedMedia.status == ProcessStatus.CREATED).count(),
        ProcessStatus.PROCESSING: query.filter(ProcessedMedia.status == ProcessStatus.PROCESSING).count(),
        ProcessStatus.DONE: query.filter(ProcessedMedia.status == ProcessStatus.DONE).count(),
        ProcessStatus.FAILED: query.filter(ProcessedMedia.status == ProcessStatus.FAILED).count(),
    }

    by_type = {}
    for row in query.with_entities(ProcessedMedia.input_type).distinct():
        by_type[row[0]] = query.filter(ProcessedMedia.input_type == row[0]).count()

    return StatsResponse(total_media=total_media, total_segments=total_segments, by_status=by_status, by_type=by_type)


@app.get("/media", response_model=MediaListResponse)
def list_media(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """List all media files."""
    # Removed user_id filtering
    query = db.query(ProcessedMedia)
    if status:
        query = query.filter(ProcessedMedia.status == status)

    total = query.count()
    items = query.order_by(ProcessedMedia.created_at.desc()).offset(skip).limit(limit).all()

    return MediaListResponse(total=total, skip=skip, limit=limit, items=[_to_response(m) for m in items])


@app.get("/media/{media_id}", response_model=MediaResponse)
def get_media(
    media_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific media file."""
    # Removed user_id filtering
    media = db.query(ProcessedMedia).filter(
        ProcessedMedia.id == media_id
    ).first()
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")
    return _to_response(media)


@app.get("/download/{media_id}")
def download_media(
    media_id: int,
    range_header: str | None = Header(None, alias="Range"),
    variant: str = Query("processed", regex="^(original|processed)$"),
    db: Session = Depends(get_db)
):
    """Download a media file."""
    # Removed user_id filtering
    media = db.query(ProcessedMedia).filter(
        ProcessedMedia.id == media_id
    ).first()
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    if variant == "processed":
        if media.status != ProcessStatus.DONE:
            raise HTTPException(status_code=400, detail=f"Processing not complete: {media.status}")

        if not media.output_path:
            raise HTTPException(status_code=404, detail="Output file not found")

        file_path = Path(media.output_path)
    else:
        # Serve original input
        if not media.input_path:
            raise HTTPException(status_code=404, detail="Input file record missing")

        file_path = Path(media.input_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{variant.capitalize()} file missing on disk")

    return range_file_response(file_path, range_header, media_type="application/octet-stream")


@app.delete("/media/{media_id}", response_model=MessageResponse)
def delete_media(
    media_id: int,
    db: Session = Depends(get_db)
):
    """Delete a media file."""
    # Removed user_id filtering
    media = db.query(ProcessedMedia).filter(
        ProcessedMedia.id == media_id
    ).first()
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    input_path = Path(media.input_path) if media.input_path else None
    output_path = Path(media.output_path) if media.output_path else None

    db.delete(media)
    db.commit()

    if input_path and input_path.exists():
        input_path.unlink()
    if output_path and output_path.exists():
        output_path.unlink()

    return MessageResponse(message="Deleted")


@app.get("/outputs/files", response_model=list[RawFileResponse])
def list_output_files(
    db: Session = Depends(get_db)
):
    """List output files."""
    # Get all output paths for all media
    all_media = db.query(ProcessedMedia).all()
    known_output_paths = {Path(m.output_path).name for m in all_media if m.output_path}
    
    files = []
    if OUTPUTS_DIR.exists():
        for path in OUTPUTS_DIR.iterdir():
            # Only list files that are known to the DB (optional, but keeps consistency)
            if path.is_file() and path.name in known_output_paths:
                # Get modification time
                stats = path.stat()
                files.append(
                    RawFileResponse(
                        filename=path.name,
                        modified_at=datetime.fromtimestamp(stats.st_mtime)
                    )
                )
    return files


@app.get("/outputs/files/{filename}")
def get_output_file(
    filename: str,
    range_header: str | None = Header(None, alias="Range"),
    db: Session = Depends(get_db)
):
    """Get an output file."""
    file_path = OUTPUTS_DIR / filename
    # Security check: prevent directory traversal and path manipulation
    try:
        resolved_path = file_path.resolve()
        resolved_outputs = OUTPUTS_DIR.resolve()
        if not str(resolved_path).startswith(str(resolved_outputs)):
            raise HTTPException(status_code=403, detail="Access denied")
        # Additional check: ensure no parent directory traversal
        if ".." in filename or filename.startswith("/") or "\\" in filename:
            raise HTTPException(status_code=403, detail="Invalid filename")
    except (ValueError, OSError):
        raise HTTPException(status_code=403, detail="Access denied")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Optional: Verify file exists in DB to prevent accessing random files if directory is shared
    target_path = str(file_path.resolve())
    media = db.query(ProcessedMedia).filter(
        ProcessedMedia.output_path == target_path
    ).first()
    
    if not media:
        # If strict correlation with DB is required, uncomment next line
        # raise HTTPException(status_code=403, detail="Access denied")
        pass

    # Determine media type based on extension
    media_type = "application/octet-stream"
    if file_path.suffix in {".mp4", ".m4v"}:
        media_type = "video/mp4"
    elif file_path.suffix in {".mp3"}:
        media_type = "audio/mpeg"
    elif file_path.suffix in {".wav"}:
        media_type = "audio/wav"

    return range_file_response(file_path, range_header, media_type=media_type)


@app.delete("/outputs/files/{filename}", response_model=MessageResponse)
def delete_output_file(
    filename: str,
    db: Session = Depends(get_db)
):
    """Delete an output file."""
    file_path = OUTPUTS_DIR / filename
    # Security check: prevent directory traversal
    try:
        resolved_path = file_path.resolve()
        resolved_outputs = OUTPUTS_DIR.resolve()
        if not str(resolved_path).startswith(str(resolved_outputs)):
            raise HTTPException(status_code=403, detail="Access denied")
        if ".." in filename or filename.startswith("/") or "\\" in filename:
            raise HTTPException(status_code=403, detail="Invalid filename")
    except (ValueError, OSError):
        raise HTTPException(status_code=403, detail="Access denied")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Resolve absolute path to match against DB
    target_path = str(file_path.resolve())

    # Find associated media record
    media = db.query(ProcessedMedia).filter(
        ProcessedMedia.output_path == target_path
    ).first()

    if media:
        input_path = Path(media.input_path) if media.input_path else None
        output_path = Path(media.output_path) if media.output_path else None

        db.delete(media)
        db.commit()

        if input_path and input_path.exists():
            input_path.unlink()
        if output_path and output_path.exists():
            output_path.unlink()
    else:
        # Just delete the orphan file
        file_path.unlink()

    return MessageResponse(message="Deleted")

    
@app.get("/debug/logs", response_class=PlainTextResponse)
def get_logs():
    """Get backend logs."""
    log_path = Path("backend.log")
    if not log_path.exists():
        return ""
    try:
        # Read file content directly
        with open(log_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return f"Error reading logs: {str(e)}"


def run_pipeline_background(media_id: int, db: Session):
    try:
        media = db.query(ProcessedMedia).filter(ProcessedMedia.id == media_id).first()
        if not media:
            return

        media.status = ProcessStatus.PROCESSING
        media.progress = 0
        media.current_activity = "Starting..."
        db.commit()

        def progress_callback(progress: int, activity: str):
            try:
                media.progress = progress
                media.current_activity = activity
                
                # Append to logs
                timestamp = utc_now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] {activity}"
                if media.logs:
                    media.logs = media.logs + "\n" + log_entry
                else:
                    media.logs = log_entry
                    
                db.commit()
            except Exception as e:
                logger.error(f"Error updating progress: {e}")

        subtitle_path = Path(media.subtitle_path) if media.subtitle_path else None

        result = process_media(
            input_path=Path(media.input_path),
            input_type=media.input_type,
            output_dir=OUTPUTS_DIR,
            filter_audio=media.filter_audio,
            filter_video=media.filter_video,
            progress_callback=progress_callback,
            subtitle_path=subtitle_path,
        )

        media.output_path = result["output_path"]
        media.status = ProcessStatus.DONE
        media.progress = 100
        media.current_activity = "Completed"
        db.commit()

        for seg in result.get("segments", []):
            db.add(CensorSegment(
                media_id=media.id,
                start_ms=int(seg["start_ms"]),
                end_ms=int(seg["end_ms"]),
                action_type=str(seg.get("action_type") or "mute"),
                reason=str(seg.get("reason") or ""),
            ))

        db.commit()

    except Exception as exc:
        logger.exception("Pipeline failed")
        try:
            media = db.query(ProcessedMedia).filter(ProcessedMedia.id == media_id).first()
            if media:
                media.status = ProcessStatus.FAILED
                media.error_message = str(exc)
                db.commit()
        except:
            pass


@app.post("/process", response_model=MediaResponse)
async def process_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subtitle_file: UploadFile = File(None),
    filter_audio: bool = Query(True),
    filter_video: bool = Query(False),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    content = await file.read()
    _validate_upload(file.filename, file.content_type, len(content))

    original_name = Path(file.filename)
    upload_path = UPLOADS_DIR / original_name.name

    counter = 1
    while upload_path.exists():
        upload_path = UPLOADS_DIR / f"{original_name.stem}_{counter}{original_name.suffix}"
        counter += 1

    upload_path.write_bytes(content)

    subtitle_path_str = None
    if subtitle_file and subtitle_file.filename:
        # Validate extension
        sub_ext = Path(subtitle_file.filename).suffix.lower()
        if sub_ext not in {".srt", ".vtt"}:
             raise HTTPException(status_code=400, detail=f"Invalid subtitle format: {sub_ext}")

        # Save subtitle file
        sub_content = await subtitle_file.read()
        sub_name = Path(subtitle_file.filename)
        sub_path = UPLOADS_DIR / f"{original_name.stem}_{sub_name.name}"

        # Ensure unique name
        sub_counter = 1
        while sub_path.exists():
             sub_path = UPLOADS_DIR / f"{original_name.stem}_{sub_counter}_{sub_name.name}"
             sub_counter += 1

        sub_path.write_bytes(sub_content)
        subtitle_path_str = str(sub_path)

    input_type = _detect_input_type(upload_path)
    file_hash = _compute_file_hash(upload_path)

    # Check for existing media with same hash and filters (Global check now, not per-user)
    existing = (
        db.query(ProcessedMedia)
        .filter(
            and_(
                ProcessedMedia.file_hash == file_hash,
                ProcessedMedia.filter_audio == filter_audio,
                ProcessedMedia.filter_video == filter_video,
                ProcessedMedia.status == ProcessStatus.DONE,
            )
        )
        .first()
    )

    if existing and existing.output_path:
        upload_path.unlink()
        return _to_response(existing)

    # Removed user_id assignment
    media = ProcessedMedia(
        input_path=str(upload_path),
        input_type=input_type,
        file_hash=file_hash,
        filter_audio=filter_audio,
        filter_video=filter_video,
        subtitle_path=subtitle_path_str,
        status=ProcessStatus.CREATED,
        progress=0,
        current_activity="Queued",
    )
    db.add(media)
    db.commit()
    db.refresh(media)

    background_tasks.add_task(run_pipeline_background, media.id, db)

    return _to_response(media)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)