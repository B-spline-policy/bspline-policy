#!/usr/bin/env python3
"""Video review UI for tidybot demos (data/demos, three-camera episodes)."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, jsonify, send_file, request

app = Flask(__name__)
DEMOS_DIR = Path(__file__).parent / "data" / "demos"
# Observation keys / filenames from episode_storage (MP4 per episode)
REAL_VIDEO_FILES = ("head_image.mp4", "left_wrist_image.mp4", "right_wrist_image.mp4")
SIM_VIDEO_FILES = ("base_image.mp4", "wrist_image.mp4")
DEMO_VIDEO_FILES = REAL_VIDEO_FILES  # default; overridden by --sim flag


def _demo_dir_complete(d: Path) -> bool:
    return d.is_dir() and all((d / name).exists() for name in DEMO_VIDEO_FILES)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Demo Review</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;500;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --accent: #00d4aa;
            --accent-dim: #00a080;
            --warning: #ffa502;
            --success: #2ed573;
            --text-primary: #e8e8ed;
            --text-secondary: #8888a0;
            --border: #2a2a3a;
        }
        
        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            background-image: 
                radial-gradient(ellipse at 20% 0%, rgba(0, 212, 170, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(0, 212, 170, 0.05) 0%, transparent 50%);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem 0;
            border-bottom: 1px solid var(--border);
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .progress-bar {
            background: var(--bg-secondary);
            border-radius: 100px;
            height: 8px;
            margin: 1.5rem auto;
            max-width: 400px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        
        .progress-fill {
            background: linear-gradient(90deg, var(--accent-dim), var(--accent));
            height: 100%;
            transition: width 0.4s ease;
            border-radius: 100px;
        }
        
        .stats {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.75rem;
        }
        
        .stats span {
            color: var(--accent);
            font-weight: 600;
        }
        
        .video-container {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .video-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }
        
        .rollout-name {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem;
            color: var(--accent);
        }
        
        .video-index {
            font-size: 0.85rem;
            color: var(--text-secondary);
            background: var(--bg-secondary);
            padding: 0.25rem 0.75rem;
            border-radius: 100px;
        }
        
        .video-grid {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1rem;
            align-items: flex-start;
            justify-content: center;
        }
        
        .video-wrapper {
            position: relative;
        }
        
        .video-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }
        
        video {
            display: block;
            height: 320px;
            width: auto;
            border-radius: 8px;
            background: var(--bg-primary);
            border: 1px solid var(--border);
        }
        
        .speed-controls {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            align-items: center;
        }
        
        .speed-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-right: 0.5rem;
        }
        
        .speed-btn {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            padding: 0.35rem 0.75rem;
            border-radius: 4px;
            border: 1px solid var(--border);
            background: var(--bg-secondary);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .speed-btn:hover {
            border-color: var(--accent);
            color: var(--text-primary);
        }
        
        .speed-btn.active {
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
        }
        
        .controls {
            display: flex;
            gap: 1rem;
            justify-content: center;
        }
        
        button {
            font-family: 'Outfit', sans-serif;
            font-size: 1rem;
            font-weight: 500;
            padding: 0.875rem 2rem;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .btn-save {
            background: var(--accent);
            color: var(--bg-primary);
        }
        
        .btn-save:hover {
            background: var(--accent-dim);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 212, 170, 0.3);
        }
        
        .btn-skip {
            background: var(--bg-secondary);
            color: var(--text-secondary);
            border: 1px solid var(--border);
        }
        
        .btn-skip:hover {
            background: var(--border);
            color: var(--text-primary);
        }
        
        .btn-success {
            background: var(--success);
            color: var(--bg-primary);
        }
        
        .btn-success:hover {
            filter: brightness(0.9);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(46, 213, 115, 0.3);
        }
        
        .btn-fail {
            background: transparent;
            color: var(--warning);
            border: 2px solid var(--warning);
        }
        
        .btn-fail:hover {
            background: var(--warning);
            color: var(--bg-primary);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 165, 2, 0.3);
        }
        
        .controls-bottom {
            display: flex;
            justify-content: center;
            margin-top: 1.25rem;
            padding-top: 1.25rem;
            border-top: 1px solid var(--border);
        }
        
        .btn-remove {
            background: transparent;
            color: #ff6b6b;
            border: 1px solid #ff6b6b;
            font-size: 0.9rem;
            padding: 0.5rem 1.25rem;
        }
        
        .btn-remove:hover {
            background: #ff6b6b;
            color: var(--bg-primary);
        }
        
        .btn-remove:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-secondary);
        }
        
        .empty-state h2 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--accent);
        }
        
        .keyboard-hint {
            margin-top: 2rem;
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        
        .keyboard-hint kbd {
            background: var(--bg-secondary);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            border: 1px solid var(--border);
            margin: 0 0.25rem;
        }
        
        .success-count {
            color: var(--success);
        }
        
        .fail-count {
            color: var(--warning);
        }
        
        @media (max-width: 768px) {
            .video-grid {
                grid-template-columns: 1fr;
            }
            
            .controls {
                flex-direction: column;
            }
            
            button {
                width: 100%;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Demo Review</h1>
            <p class="subtitle">Count success and fail demos in data/demos</p>
            
            <div class="progress-bar">
                <div class="progress-fill" id="progress" style="width: 0%"></div>
            </div>
            <p class="stats">
                <span id="current">0</span> / <span id="total">0</span> reviewed
                · <span id="success" class="success-count">0</span> success
                · <span id="fail" class="fail-count">0</span> fail
            </p>
        </header>
        
        <div id="content">
            <div class="empty-state">
                <h2>Loading...</h2>
                <p>Scanning demos directory</p>
            </div>
        </div>
        
        <div class="keyboard-hint">
            Keyboard shortcuts: <kbd>Y</kbd> Success · <kbd>N</kbd> Fail · <kbd>S</kbd> Skip · <kbd>R</kbd> Remove · Speed: <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> <kbd>4</kbd>
        </div>
    </div>
    
    <script>
        let demos = [];
        let videoFiles = [];
        let currentIndex = 0;
        let successCount = 0;
        let failCount = 0;
        let successFiles = [];
        let failFiles = [];
        let currentSpeed = 1;
        const speeds = [0.5, 1, 2, 4];

        async function loadDemos() {
            const [demosResp, vfResp] = await Promise.all([
                fetch('/api/demos'),
                fetch('/api/video_files'),
            ]);
            demos = await demosResp.json();
            videoFiles = await vfResp.json();
            document.getElementById('total').textContent = demos.length;
            if (demos.length > 0) {
                showDemo(0);
            } else {
                showEmpty();
            }
        }
        
        function setSpeed(speed) {
            currentSpeed = speed;
            const videos = document.querySelectorAll('video');
            videos.forEach(v => v.playbackRate = speed);
            
            // Update speed button styles
            document.querySelectorAll('.speed-btn').forEach(btn => {
                btn.classList.toggle('active', parseFloat(btn.dataset.speed) === speed);
            });
        }
        
        function getSpeedControlsHTML() {
            return `
                <div class="speed-controls">
                    <span class="speed-label">Speed:</span>
                    ${speeds.map(s => `
                        <button class="speed-btn ${s === currentSpeed ? 'active' : ''}" 
                                data-speed="${s}" 
                                onclick="setSpeed(${s})">${s}x</button>
                    `).join('')}
                </div>
            `;
        }
        
        function getControlsHTML() {
            return `
                <div class="controls">
                    <button class="btn-success" onclick="markSuccess()">
                        ✓ Success
                    </button>
                    <button class="btn-skip" onclick="skipDemo()">
                        → Skip
                    </button>
                    <button class="btn-fail" onclick="markFail()">
                        ✗ Fail
                    </button>
                </div>
                <div class="controls-bottom">
                    <button type="button" class="btn-remove" onclick="removeDemo()">
                        Remove
                    </button>
                </div>
            `;
        }
        
        function showDemo(index) {
            if (index >= demos.length) {
                showComplete();
                return;
            }
            
            currentIndex = index;
            const demoId = demos[index];
            
            document.getElementById('current').textContent = index + 1;
            document.getElementById('progress').style.width = 
                ((index + 1) / demos.length * 100) + '%';
            
            document.getElementById('content').innerHTML = `
                <div class="video-container">
                    <div class="video-header">
                        <span class="rollout-name">${demoId}</span>
                        <span class="video-index">${index + 1} of ${demos.length}</span>
                    </div>
                    <div class="video-grid">
                        ${videoFiles.map(f => {
                            const label = f.replace('_image.mp4', '').replace(/_/g, ' ');
                            return `
                            <div class="video-wrapper">
                                <div class="video-label">${label}</div>
                                <video controls autoplay loop muted>
                                    <source src="/video/${demoId}/${f}" type="video/mp4">
                                </video>
                            </div>`;
                        }).join('')}
                    </div>
                    ${getSpeedControlsHTML()}
                    ${getControlsHTML()}
                </div>
            `;
            
            // Apply current speed to new videos
            setTimeout(() => setSpeed(currentSpeed), 100);
        }
        
        function showEmpty() {
            document.getElementById('content').innerHTML = `
                <div class="empty-state">
                    <h2>No Demos Found</h2>
                    <p>The demos directory is empty or episodes are missing the three camera MP4s.</p>
                </div>
            `;
        }
        
        function showComplete() {
            document.getElementById('progress').style.width = '100%';
            
            const total = successCount + failCount;
            const successRate = total > 0 ? ((successCount / total) * 100).toFixed(1) : 0;
            document.getElementById('content').innerHTML = `
                <div class="empty-state">
                    <h2>Counting Complete! 📊</h2>
                    <p>You've reviewed all ${demos.length} demos.</p>
                    <p style="margin-top: 1rem;">
                        <span class="success-count">${successCount}</span> success · 
                        <span class="fail-count">${failCount}</span> fail
                    </p>
                    <p style="margin-top: 0.5rem; font-size: 1.25rem;">
                        Success Rate: <span class="success-count">${successRate}%</span>
                    </p>
                    <div style="margin-top: 1.5rem;">
                        <button class="btn-save" onclick="saveResults()" id="save-btn">
                            💾 Save Results to JSON
                        </button>
                    </div>
                    <p id="save-status" style="margin-top: 1rem; font-size: 0.9rem;"></p>
                </div>
            `;
        }
        
        async function saveResults() {
            const btn = document.getElementById('save-btn');
            const status = document.getElementById('save-status');
            
            btn.disabled = true;
            btn.textContent = '⏳ Saving...';
            
            const data = {
                timestamp: new Date().toISOString(),
                total_reviewed: successCount + failCount,
                success_count: successCount,
                fail_count: failCount,
                success_rate: successCount + failCount > 0 
                    ? ((successCount / (successCount + failCount)) * 100).toFixed(1) + '%' 
                    : '0%',
                success_files: successFiles,
                fail_files: failFiles
            };
            
            try {
                const response = await fetch('/api/save_results', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    btn.textContent = '✓ Saved!';
                    btn.style.background = 'var(--success)';
                    status.innerHTML = `Saved to: <code style="background: var(--bg-secondary); padding: 0.25rem 0.5rem; border-radius: 4px;">${result.filepath}</code>`;
                    status.style.color = 'var(--success)';
                } else {
                    throw new Error(result.error || 'Failed to save');
                }
            } catch (err) {
                btn.textContent = '✕ Error';
                btn.style.background = 'var(--danger)';
                status.textContent = 'Failed to save: ' + err.message;
                status.style.color = 'var(--danger)';
                btn.disabled = false;
                setTimeout(() => {
                    btn.textContent = '💾 Save Results to JSON';
                    btn.style.background = 'var(--accent)';
                }, 2000);
            }
        }
        
        function markSuccess() {
            const demoId = demos[currentIndex];
            successFiles.push(demoId);
            successCount++;
            document.getElementById('success').textContent = successCount;
            showDemo(currentIndex + 1);
        }
        
        function markFail() {
            const demoId = demos[currentIndex];
            failFiles.push(demoId);
            failCount++;
            document.getElementById('fail').textContent = failCount;
            showDemo(currentIndex + 1);
        }
        
        function skipDemo() {
            showDemo(currentIndex + 1);
        }
        
        async function removeDemo() {
            if (demos.length === 0 || currentIndex >= demos.length) return;
            const demoId = demos[currentIndex];
            if (!confirm(`Delete episode folder "${demoId}" from disk? This cannot be undone.`)) return;
            
            const btn = document.querySelector('.btn-remove');
            if (btn) {
                btn.disabled = true;
            }
            try {
                const response = await fetch('/api/remove_demo', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ demo_id: demoId })
                });
                const result = await response.json();
                if (!response.ok) {
                    throw new Error(result.error || 'Remove failed');
                }
                demos.splice(currentIndex, 1);
                document.getElementById('total').textContent = demos.length;
                if (demos.length === 0) {
                    document.getElementById('current').textContent = '0';
                    document.getElementById('progress').style.width = '0%';
                    showEmpty();
                    return;
                }
                const nextIdx = Math.min(currentIndex, demos.length - 1);
                showDemo(nextIdx);
            } catch (err) {
                alert(err.message);
                if (btn) btn.disabled = false;
            }
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            
            const key = e.key.toLowerCase();
            
            // Speed controls (1-4 keys)
            if (key >= '1' && key <= '4') {
                const speedIndex = parseInt(key) - 1;
                if (speedIndex < speeds.length) {
                    setSpeed(speeds[speedIndex]);
                }
                return;
            }
            
            switch(key) {
                case 'y':
                    markSuccess();
                    break;
                case 'n':
                    markFail();
                    break;
                case 's':
                    skipDemo();
                    break;
                case 'r':
                    removeDemo();
                    break;
            }
        });
        
        loadDemos();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/demos')
def get_demos():
    """List episode dirs under data/demos that have all camera MP4s."""
    if not DEMOS_DIR.exists():
        return jsonify([])

    demo_ids = sorted(d.name for d in DEMOS_DIR.iterdir() if _demo_dir_complete(d))
    return jsonify(demo_ids)


@app.route('/api/video_files')
def get_video_files():
    """Return the list of video filenames for the current mode."""
    return jsonify(list(DEMO_VIDEO_FILES))


@app.route('/video/<demo_id>/<filename>')
def serve_video(demo_id, filename):
    """Serve a video file."""
    video_path = DEMOS_DIR / demo_id / filename
    if not video_path.exists():
        return "Video not found", 404
    return send_file(video_path, mimetype='video/mp4')


def _safe_demo_path(demo_id: str) -> Path | None:
    if not demo_id or demo_id != Path(demo_id).name:
        return None
    if ".." in demo_id:
        return None
    p = (DEMOS_DIR / demo_id).resolve()
    try:
        p.relative_to(DEMOS_DIR.resolve())
    except ValueError:
        return None
    return p


@app.route('/api/remove_demo', methods=['POST'])
def remove_demo():
    """Delete an episode directory under data/demos."""
    payload = request.get_json(silent=True) or {}
    demo_id = payload.get("demo_id")
    if not isinstance(demo_id, str):
        return jsonify({"error": "demo_id required"}), 400
    target = _safe_demo_path(demo_id)
    if target is None or not target.is_dir():
        return jsonify({"error": "Invalid or missing demo"}), 404
    try:
        shutil.rmtree(target)
        return jsonify({"success": True, "removed": demo_id})
    except OSError as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/save_results', methods=['POST'])
def save_results():
    """Save counting results to a JSON file."""
    data = request.get_json()
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"review_results_{timestamp}.json"
    filepath = DEMOS_DIR / filename
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return jsonify({"success": True, "filepath": str(filepath)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Demo Review UI')
    parser.add_argument('--sim', action='store_true', help='Use sim camera names (base, wrist)')
    args = parser.parse_args()

    if args.sim:
        DEMO_VIDEO_FILES = SIM_VIDEO_FILES

    mode = "sim" if args.sim else "real"
    print(f"\n🎬 Demo Review UI ({mode})")
    print(f"📁 Scanning: {DEMOS_DIR}")
    print(f"📷 Expecting: {', '.join(DEMO_VIDEO_FILES)}")

    demo_count = len([d for d in DEMOS_DIR.iterdir() if _demo_dir_complete(d)]) if DEMOS_DIR.exists() else 0
    print(f"📊 Found {demo_count} demos\n")

    print("🌐 Open http://localhost:5050 in your browser\n")
    app.run(host='0.0.0.0', port=5050, debug=False)
