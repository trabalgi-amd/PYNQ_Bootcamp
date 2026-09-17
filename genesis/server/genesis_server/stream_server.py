"""
Web-based live viewer for Genesis simulations.

This module provides a Flask-based streaming server that runs alongside
the main Genesis server, allowing students to view their robot simulations
in real-time via a web browser.
"""

import base64
import time
import threading
from typing import Dict, Optional
from flask import Flask, Response, render_template_string, jsonify
import numpy as np

# Will be set by the main server
simulations: Dict[str, 'GenesisSimulation'] = {}
simulations_lock: threading.Lock = None  # Protects simulations dict access

# Track active stream viewers per token to manage camera recording state
_stream_viewer_counts: Dict[str, int] = {}
_viewer_counts_lock = threading.Lock()

app = Flask(__name__)
app.config['THREADED'] = True


def _with_lock(func):
    """Execute func while holding the simulations lock, if available."""
    if simulations_lock is not None:
        with simulations_lock:
            return func()
    return func()


def _add_viewer(token: str, sim) -> bool:
    """Register a new stream viewer for token. Returns True if this is the first viewer.

    Recording is started in generate_frames() directly, not here.
    """
    with _viewer_counts_lock:
        prev_count = _stream_viewer_counts.get(token, 0)
        _stream_viewer_counts[token] = prev_count + 1
        is_first = prev_count == 0

    return is_first


def _remove_viewer(token: str, sim) -> bool:
    """Unregister a stream viewer for token. Returns True if this was the last viewer.

    If last viewer, stops camera recording mode.
    """
    with _viewer_counts_lock:
        count = _stream_viewer_counts.get(token, 0)
        if count <= 1:
            _stream_viewer_counts.pop(token, None)
            is_last = True
        else:
            _stream_viewer_counts[token] = count - 1
            is_last = False

    if is_last and sim and sim.camera:
        try:
            sim.camera.pause_recording()
        except Exception:
            pass  # Ignore - sim may be shutting down

    return is_last


def cleanup_viewer_state(token: str) -> None:
    """Clean up viewer tracking state for a destroyed simulation.

    Called by the main server when a simulation is destroyed to ensure
    viewer counts don't leak.
    """
    with _viewer_counts_lock:
        _stream_viewer_counts.pop(token, None)


def _generate_message_frame(message: str, color: tuple = (100, 100, 100)) -> bytes:
    """Generate a JPEG frame with a centered text message.

    Args:
        message: Text to display
        color: RGB tuple for text color

    Returns:
        JPEG image bytes (not wrapped in MJPEG frame format)
    """
    import io
    from PIL import Image, ImageDraw

    img = Image.new('RGB', (640, 480), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Center the text
    text_bbox = draw.textbbox((0, 0), message)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (640 - text_width) // 2
    y = (480 - text_height) // 2

    draw.text((x, y), message, fill=color)

    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()


def stop_all_recordings() -> None:
    """Stop all camera recordings before server shutdown.

    Clears recording buffers to prevent Genesis from saving video files on exit.
    """
    with _viewer_counts_lock:
        tokens = list(_stream_viewer_counts.keys())
        _stream_viewer_counts.clear()

    def _stop_recordings():
        for token in tokens:
            sim = simulations.get(token)
            if sim and sim.camera:
                try:
                    sim.camera.pause_recording()
                    # Clear the recorded frames buffer to prevent video save
                    if hasattr(sim.camera, '_recorded_imgs'):
                        sim.camera._recorded_imgs = []
                except Exception:
                    pass

        # Also check all simulations in case any were missed
        for sim in simulations.values():
            if sim and sim.camera:
                try:
                    sim.camera.pause_recording()
                    if hasattr(sim.camera, '_recorded_imgs'):
                        sim.camera._recorded_imgs = []
                except Exception:
                    pass

    _with_lock(_stop_recordings)


def cleanup_orphaned_viewer_counts() -> int:
    """Remove viewer counts for tokens that no longer have active simulations.

    Returns the number of orphaned entries cleaned up.
    """
    with _viewer_counts_lock:
        # Get tokens that have viewer counts
        tracked_tokens = list(_stream_viewer_counts.keys())

    if not tracked_tokens:
        return 0

    # Check which tokens still have active simulations
    def _get_active_tokens():
        return set(simulations.keys())

    active_tokens = _with_lock(_get_active_tokens)

    # Find orphaned tokens
    orphaned = [t for t in tracked_tokens if t not in active_tokens]

    # Clean up orphaned entries
    with _viewer_counts_lock:
        for token in orphaned:
            _stream_viewer_counts.pop(token, None)

    if orphaned:
        print(f"Cleaned up {len(orphaned)} orphaned viewer count(s)")

    return len(orphaned)


def _get_sim_snapshot(token):
    """Get simulation and its state atomically under lock.

    Returns a tuple of (sim, initialized, scene_name, num_robots, session_id)
    or (None, False, None, 0, "?") if not found.
    """
    def _snapshot():
        sim = simulations.get(token)
        if sim is None:
            return (None, False, None, 0, "?")
        sorted_tokens = sorted(simulations.keys())
        session_id = sorted_tokens.index(token) + 1 if token in sorted_tokens else "?"
        return (sim, sim._initialized, sim.scene_name, len(sim.robots), session_id)
    return _with_lock(_snapshot)


VIEWER_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robot Simulation - Live View</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            text-align: center;
            border-bottom: 3px solid #4CAF50;
        }

        .header h1 {
            font-size: 32px;
            margin-bottom: 5px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }

        .header .subtitle {
            font-size: 14px;
            opacity: 0.8;
        }

        .container {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .video-container {
            background: rgba(0, 0, 0, 0.5);
            border: 5px solid #4CAF50;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
            max-width: 95%;
            max-height: 70vh;
        }

        .video-container img {
            display: block;
            width: 100%;
            height: auto;
        }

        .info-panel {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 15px 25px;
            margin-top: 20px;
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            justify-content: center;
        }

        .info-item {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .info-item .label {
            font-size: 12px;
            opacity: 0.7;
            margin-bottom: 5px;
        }

        .info-item .value {
            font-size: 18px;
            font-weight: bold;
            color: #4CAF50;
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            display: inline-block;
            margin-right: 5px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .instructions {
            background: rgba(255, 255, 255, 0.1);
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin-top: 20px;
            border-radius: 5px;
            max-width: 600px;
        }

        .instructions h3 {
            margin-bottom: 10px;
            font-size: 16px;
        }

        .instructions p {
            font-size: 14px;
            line-height: 1.6;
            opacity: 0.9;
        }

        .error-container {
            background: rgba(244, 67, 54, 0.2);
            border: 2px solid #f44336;
            border-radius: 10px;
            padding: 30px;
            max-width: 600px;
            text-align: center;
        }

        .error-container h2 {
            color: #f44336;
            margin-bottom: 15px;
        }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 24px;
            }

            .info-panel {
                gap: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Robot Simulation - Live View</h1>
        <p class="subtitle">Session: {{ token[:8] }}...</p>
    </div>

    <div class="container">
        {% if error %}
        <div class="error-container">
            <h2>⚠️ {{ error_title }}</h2>
            <p>{{ error_message }}</p>
        </div>
        {% else %}
        <div class="video-container">
            <img src="{{ url_for('stream_video', token=token) }}"
                 alt="Robot simulation live stream"
                 id="video-stream">
        </div>

        <div class="info-panel">
            <div class="info-item">
                <span class="label">Status</span>
                <span class="value">
                    <span class="status-indicator"></span>
                    Live
                </span>
            </div>
            <div class="info-item">
                <span class="label">Scene</span>
                <span class="value">{{ scene_name }}</span>
            </div>
            <div class="info-item">
                <span class="label">Robots</span>
                <span class="value">{{ num_robots }}</span>
            </div>
        </div>

        <div class="instructions">
            <h3>💡 Tips</h3>
            <p>
                Keep this tab open while you code in Jupyter!
                You'll see your robot move in real-time as you run commands.
                The stream updates automatically at ~30 FPS.
            </p>
        </div>
        {% endif %}
    </div>

    <script>
        // Detect if stream fails to load
        const img = document.getElementById('video-stream');
        if (img) {
            img.onerror = function() {
                document.querySelector('.video-container').innerHTML =
                    '<div style="padding: 40px; text-align: center;">' +
                    '<h3 style="color: #f44336;">Stream Unavailable</h3>' +
                    '<p style="opacity: 0.8;">The simulation may have ended or the server restarted.</p>' +
                    '<p style="margin-top: 10px;"><a href="" style="color: #4CAF50;">Refresh Page</a></p>' +
                    '</div>';
            };
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Landing page showing available streams."""
    # Thread-safe snapshot of active sessions
    def _get_active_sessions():
        active = []
        sorted_tokens = sorted(simulations.keys())
        for idx, token in enumerate(sorted_tokens, start=1):
            sim = simulations.get(token)
            if sim and sim._initialized:
                active.append({
                    'token': token,
                    'token_short': token[:8],
                    'scene': sim.scene_name,
                    'num_robots': len(sim.robots),
                    'session_id': idx
                })
        return active

    active_sessions = _with_lock(_get_active_sessions)

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genesis Live Viewer</title>
        <style>
            body {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                font-family: Arial, sans-serif;
                color: white;
                padding: 40px;
            }
            h1 { text-align: center; margin-bottom: 30px; }
            .sessions {
                max-width: 800px;
                margin: 0 auto;
            }
            .session-card {
                background: rgba(0, 0, 0, 0.3);
                padding: 20px;
                margin: 15px 0;
                border-radius: 10px;
                border-left: 5px solid #4CAF50;
            }
            .session-card a {
                color: #4CAF50;
                text-decoration: none;
                font-size: 18px;
            }
            .empty {
                text-align: center;
                opacity: 0.7;
            }
        </style>
    </head>
    <body>
        <h1>🤖 Genesis Live Viewer</h1>
        <div class="sessions">
    """

    if active_sessions:
        for session in active_sessions:
            html += f"""
            <div class="session-card">
                <a href="/view/{session['token']}">
                    Session {session['token_short']} - {session['scene']}
                </a>
                <div style="font-size: 14px; opacity: 0.8; margin-top: 5px;">
                    Robots: {session['num_robots']}
                </div>
            </div>
            """
    else:
        html += '<p class="empty">No active sessions. Create an environment to start streaming!</p>'

    html += """
        </div>
    </body>
    </html>
    """

    return html


@app.route('/view/<token>')
def view_simulation(token):
    """Viewer page for a specific simulation."""
    # Thread-safe snapshot of simulation state
    sim, initialized, scene_name, num_robots, session_id = _get_sim_snapshot(token)

    if not sim:
        return render_template_string(
            VIEWER_TEMPLATE,
            token=token,
            error=True,
            error_title="Session Not Found",
            error_message="No active simulation found for this session. The session may have expired or been destroyed."
        )

    if not initialized:
        return render_template_string(
            VIEWER_TEMPLATE,
            token=token,
            error=True,
            error_title="Simulation Initializing",
            error_message="The simulation is still starting up. Please refresh in a moment."
        )

    return render_template_string(
        VIEWER_TEMPLATE,
        token=token,
        scene_name=scene_name,
        num_robots=num_robots,
        error=False
    )


@app.route('/stream/<token>')
def stream_video(token):
    """MJPEG stream endpoint for a specific simulation."""
    # Thread-safe check if simulation exists and is initialized
    sim, initialized, _, _, _ = _get_sim_snapshot(token)

    if not sim or not initialized:
        # Return a placeholder error image
        def generate_error():
            frame_data = _generate_message_frame("Session Not Found", color=(255, 100, 100))
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        return Response(generate_error(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_frames():
        """Generate frames from the simulation camera."""
        frame_delay = 1.0 / 30.0  # Target 30 FPS
        stream_ended_normally = False

        # Thread-safe check and get simulation reference
        sim, initialized, _, _, _ = _get_sim_snapshot(token)
        if not sim or not initialized or not sim.camera:
            return

        # Register this viewer for tracking
        _add_viewer(token, sim)

        # Always ensure recording is started (idempotent - safe to call multiple times)
        try:
            sim.camera.start_recording()
        except Exception as e:
            print(f"[Stream] start_recording error for {token[:8]}: {e}")

        # Track the last valid sim reference for cleanup
        last_valid_sim = sim
        frame_count = 0

        try:
            while True:
                try:
                    # Thread-safe check if simulation still exists
                    current_sim, current_initialized, _, _, _ = _get_sim_snapshot(token)
                    if not current_sim or not current_initialized:
                        # Simulation was destroyed - send a final "ended" frame
                        stream_ended_normally = True
                        break

                    sim = current_sim
                    last_valid_sim = sim  # Update last valid reference

                    if sim.camera:
                        # Render current frame
                        sim.camera.render()

                        # Debug: log recording state periodically
                        if frame_count == 0:
                            in_rec = getattr(sim.camera, '_in_recording', 'unknown')
                            has_imgs = hasattr(sim.camera, '_recorded_imgs')
                            img_count = len(sim.camera._recorded_imgs) if has_imgs else 0
                            print(f"[Stream] {token[:8]}: in_recording={in_rec}, has_imgs={has_imgs}, count={img_count}")

                        # Access the recorded images buffer
                        if hasattr(sim.camera, '_recorded_imgs') and len(sim.camera._recorded_imgs) > 0:
                            # Get the most recent frame
                            rgba = sim.camera._recorded_imgs[-1]

                                # Clear old frames to prevent memory buildup (but keep a few for other viewers)
                            if len(sim.camera._recorded_imgs) > 10:
                                sim.camera._recorded_imgs = sim.camera._recorded_imgs[-5:]

                            frame_count += 1
                        else:
                            time.sleep(frame_delay)
                            continue

                        # Convert RGBA to RGB
                        if len(rgba.shape) == 3 and rgba.shape[2] == 4:
                            rgb = rgba[:, :, :3]
                        else:
                            rgb = rgba

                        # Convert to uint8 if needed
                        if rgb.dtype != np.uint8:
                            if rgb.max() <= 1.0:
                                rgb = (rgb * 255).astype(np.uint8)
                            else:
                                rgb = rgb.astype(np.uint8)

                        # Encode as JPEG
                        import cv2
                        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                                 [cv2.IMWRITE_JPEG_QUALITY, 85])

                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

                        time.sleep(frame_delay)
                    else:
                        time.sleep(frame_delay)
                        continue

                except Exception as e:
                    # Suppress common shutdown-related errors
                    err_str = str(e)
                    if 'UID' in err_str or 'shutdown' in err_str.lower() or 'closed' in err_str.lower():
                        # Server is shutting down, exit gracefully
                        break
                    print(f"Stream error for {token[:8]}: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1.0)
                    continue

            # Send a final "stream ended" frame so the browser shows a message
            if stream_ended_normally:
                ended_frame = _generate_message_frame("Session Ended - Refresh to Reconnect", color=(255, 200, 100))
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + ended_frame + b'\r\n')

        finally:
            # Unregister this viewer (stops recording if last viewer)
            # Use last_valid_sim to ensure we call pause_recording on the right object
            _remove_viewer(token, last_valid_sim)

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot/<token>')
def get_snapshot(token):
    """Get a single snapshot (for debugging or fallback)."""
    # Thread-safe check if simulation exists
    sim, initialized, _, _, _ = _get_sim_snapshot(token)

    if not sim or not initialized or not sim.camera:
        return jsonify({'error': 'Session not found'}), 404

    try:
        # Check if there are active stream viewers (recording already started)
        with _viewer_counts_lock:
            has_active_viewers = _stream_viewer_counts.get(token, 0) > 0

        # Start recording temporarily if no active viewers
        if not has_active_viewers:
            sim.camera.start_recording()

        sim.camera.render()

        # Get frame from recorded images buffer
        if hasattr(sim.camera, '_recorded_imgs') and len(sim.camera._recorded_imgs) > 0:
            rgba = sim.camera._recorded_imgs[-1]
        else:
            if not has_active_viewers:
                sim.camera.pause_recording()
            return jsonify({'error': 'No frame available'}), 500

        # Stop recording if we started it (no active viewers)
        if not has_active_viewers:
            sim.camera.pause_recording()

        # Convert RGBA to RGB
        if len(rgba.shape) == 3 and rgba.shape[2] == 4:
            rgb = rgba[:, :, :3]
        else:
            rgb = rgba

        # Convert to uint8 if needed
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        import cv2
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                 [cv2.IMWRITE_JPEG_QUALITY, 90])

        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': img_base64,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint."""
    # Thread-safe count of sessions
    def _count_sessions():
        active = sum(1 for s in simulations.values() if s._initialized)
        total = len(simulations)
        return active, total

    active_sessions, total_sessions = _with_lock(_count_sessions)

    return jsonify({
        'status': 'ok',
        'active_sessions': active_sessions,
        'total_sessions': total_sessions
    })


def start_stream_server(port: int = 9003, simulations_dict: Dict = None, lock: threading.Lock = None):
    """
    Start the Flask streaming server in a background thread.

    Args:
        port: Port to run the stream server on (default: 9003)
        simulations_dict: Reference to the main server's simulations dictionary
        lock: Threading lock to protect simulations dict access
    """
    global simulations, simulations_lock

    if simulations_dict is not None:
        simulations = simulations_dict
    if lock is not None:
        simulations_lock = lock

    def run_server():
        # Suppress Flask's startup messages for cleaner output
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        print(f"Open live view at port {port}")

        app.run(
            host='0.0.0.0',
            port=port,
            threaded=True,
            debug=False,
            use_reloader=False  # Important: disable reloader in thread
        )

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    return thread
