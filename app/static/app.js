let cameraRunning = false;
let detectionRunning = false;

// --- Camera ---

async function loadCameras() {
    const select = document.getElementById('camera-select');
    try {
        const res = await fetch('/api/camera/list');
        const cameras = await res.json();
        select.innerHTML = '<option value="">Default Camera</option>';
        cameras.forEach(cam => {
            const opt = document.createElement('option');
            opt.value = cam.index;
            opt.textContent = cam.name;
            select.appendChild(opt);
        });
    } catch (e) {
        console.warn('Could not enumerate cameras:', e);
    }
}

async function toggleCamera() {
    const btn = document.getElementById('btn-camera');
    const detectBtn = document.getElementById('btn-detect');
    const cameraSelect = document.getElementById('camera-select');

    if (!cameraRunning) {
        btn.textContent = 'Starting...';
        btn.disabled = true;
        const idx = cameraSelect.value;
        const url = idx !== '' ? `/api/camera/start?camera_index=${idx}` : '/api/camera/start';
        const res = await fetch(url, { method: 'POST' });
        if (res.ok) {
            cameraRunning = true;
            btn.textContent = 'Stop Camera';
            btn.classList.add('active');
            btn.disabled = false;
            detectBtn.disabled = false;
            cameraSelect.disabled = true;
            document.getElementById('btn-snap').disabled = false;
            showFeed();
            updateStatus();
        } else {
            const data = await res.json();
            alert(data.detail || 'Failed to start camera');
            btn.textContent = 'Start Camera';
            btn.disabled = false;
        }
    } else {
        if (detectionRunning) {
            await toggleDetection();
        }
        await fetch('/api/camera/stop', { method: 'POST' });
        cameraRunning = false;
        btn.textContent = 'Start Camera';
        btn.classList.remove('active');
        detectBtn.disabled = true;
        cameraSelect.disabled = false;
        document.getElementById('btn-snap').disabled = true;
        hideFeed();
        updateStatus();
    }
}

function showFeed() {
    const feed = document.getElementById('camera-feed');
    const placeholder = document.getElementById('feed-placeholder');
    feed.src = '/api/camera/feed?' + Date.now();
    feed.classList.add('active');
    placeholder.style.display = 'none';
}

function hideFeed() {
    const feed = document.getElementById('camera-feed');
    const placeholder = document.getElementById('feed-placeholder');
    feed.src = '';
    feed.classList.remove('active');
    placeholder.style.display = 'flex';
}

// --- Detection ---

async function toggleDetection() {
    const btn = document.getElementById('btn-detect');

    if (!detectionRunning) {
        btn.textContent = 'Starting...';
        btn.disabled = true;
        const res = await fetch('/api/detection/start', { method: 'POST' });
        if (res.ok) {
            detectionRunning = true;
            btn.textContent = 'Stop Detection';
            btn.classList.add('active');
            btn.disabled = false;
            updateStatus();
        } else {
            const data = await res.json();
            alert(data.detail || 'Failed to start detection');
            btn.textContent = 'Start Detection';
            btn.disabled = false;
        }
    } else {
        await fetch('/api/detection/stop', { method: 'POST' });
        detectionRunning = false;
        btn.textContent = 'Start Detection';
        btn.classList.remove('active');
        updateStatus();
    }
}

function updateStatus() {
    const indicator = document.getElementById('status-indicator');
    if (detectionRunning) {
        indicator.textContent = 'Detecting';
        indicator.className = 'status detecting';
    } else if (cameraRunning) {
        indicator.textContent = 'Camera On';
        indicator.className = 'status camera-on';
    } else {
        indicator.textContent = 'Idle';
        indicator.className = 'status off';
    }
}

// --- Search ---

async function searchItem(e) {
    e.preventDefault();
    const query = document.getElementById('search-input').value.trim();
    if (!query) return;

    const resultsDiv = document.getElementById('search-results');
    resultsDiv.innerHTML = '<p style="color:#666">Searching...</p>';

    const res = await fetch(`/api/search/?q=${encodeURIComponent(query)}`);
    const data = await res.json();

    if (!data.length) {
        resultsDiv.innerHTML = '<p class="no-results">No sightings found for that item. Make sure it\'s registered and detection is running.</p>';
        return;
    }

    resultsDiv.innerHTML = data.map(r => `
        <div class="result-card">
            <img src="${r.image_url}" alt="${r.item_name}">
            <div class="result-info">
                <h3>${r.item_name}</h3>
                <p>Last seen: ${formatTime(r.last_seen)}</p>
                <p class="confidence">Confidence: ${(r.similarity * 100).toFixed(1)}%</p>
            </div>
        </div>
    `).join('');
}

function formatTime(ts) {
    const d = new Date(ts + 'Z');
    return d.toLocaleString();
}

// --- Snapshot for registration ---

let _snapshotBlobs = [];

async function takeSnapshot() {
    const btn = document.getElementById('btn-snap');
    const previewsDiv = document.getElementById('snap-previews');
    const photoInput = document.getElementById('item-photo');
    const statusDiv = document.getElementById('register-status');

    btn.textContent = 'Capturing...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/camera/snapshot');
        if (!res.ok) throw new Error('Camera not ready');
        const blob = await res.blob();
        _snapshotBlobs.push(blob);

        // Clear any file selection — snapshots take precedence
        photoInput.value = '';

        // Add preview thumbnail
        const url = URL.createObjectURL(blob);
        const thumb = document.createElement('div');
        thumb.className = 'snap-thumb';
        thumb.innerHTML = `<img src="${url}" alt="Snapshot ${_snapshotBlobs.length}">
            <span class="snap-num">${_snapshotBlobs.length}</span>`;
        previewsDiv.appendChild(thumb);

        const count = _snapshotBlobs.length;
        statusDiv.innerHTML = `<span style="color:#00d4ff">${count} photo${count > 1 ? 's' : ''} captured. Take more or click Register.</span>`;
    } catch (err) {
        statusDiv.innerHTML = `<span class="error">Could not capture: ${err.message}</span>`;
    }

    btn.textContent = 'Take Another Photo';
    btn.disabled = false;
}

// --- Registration mode toggle ---

function toggleRegMode() {
    const mode = document.querySelector('input[name="reg-mode"]:checked').value;
    const photoRow = document.getElementById('photo-input-row');
    const previewsDiv = document.getElementById('snap-previews');
    if (mode === 'photo') {
        photoRow.style.display = '';
    } else {
        photoRow.style.display = 'none';
        previewsDiv.innerHTML = '';
        _snapshotBlobs = [];
    }
}

// --- Register ---

async function registerItem(e) {
    e.preventDefault();
    const nameInput = document.getElementById('item-name');
    const photoInput = document.getElementById('item-photo');
    const previewsDiv = document.getElementById('snap-previews');
    const statusDiv = document.getElementById('register-status');
    const mode = document.querySelector('input[name="reg-mode"]:checked').value;

    const name = nameInput.value.trim();
    if (!name) return;

    if (mode === 'text') {
        // Text-only registration
        statusDiv.innerHTML = '<span style="color:#666">Registering... (generating text embeddings)</span>';
        const res = await fetch('/api/items/register-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        const data = await res.json();
        if (res.ok) {
            statusDiv.innerHTML = `<span class="success">Registered "${data.name}" via text description!</span>`;
            nameInput.value = '';
            loadItems();
        } else {
            let errMsg = 'Registration failed';
            if (data.detail) {
                errMsg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
            }
            statusDiv.innerHTML = `<span class="error">${errMsg}</span>`;
        }
        return;
    }

    // Photo registration
    const form = new FormData();
    form.append('name', name);

    // Determine photo source: snapshot blobs > file input
    if (_snapshotBlobs.length > 0) {
        _snapshotBlobs.forEach((blob, i) => {
            form.append('photos', new File([blob], `snapshot_${i + 1}.jpg`, { type: 'image/jpeg' }));
        });
    } else if (photoInput.files.length > 0) {
        for (const file of photoInput.files) {
            form.append('photos', file);
        }
    } else {
        statusDiv.innerHTML = '<span class="error">Please upload photo(s) or take them with the camera.</span>';
        return;
    }

    statusDiv.innerHTML = '<span style="color:#666">Registering... (generating CLIP embeddings)</span>';

    const res = await fetch('/api/items/register', { method: 'POST', body: form });
    const data = await res.json();

    if (res.ok) {
        statusDiv.innerHTML = `<span class="success">Registered "${data.name}" with ${data.photo_count} photo(s)!</span>`;
        nameInput.value = '';
        photoInput.value = '';
        _snapshotBlobs = [];
        previewsDiv.innerHTML = '';
        document.getElementById('btn-snap').textContent = 'Take Photo';
        loadItems();
    } else {
        let errMsg = 'Registration failed';
        if (data.detail) {
            errMsg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
        }
        statusDiv.innerHTML = `<span class="error">${errMsg}</span>`;
    }
}

// --- Add photo to existing item ---

async function addPhotoToItem(itemId, itemName) {
    if (!cameraRunning) {
        alert('Start the camera first to take a photo.');
        return;
    }

    const res = await fetch('/api/camera/snapshot');
    if (!res.ok) {
        alert('Could not capture photo. Is the camera running?');
        return;
    }
    const blob = await res.blob();
    const form = new FormData();
    form.append('photo', new File([blob], 'additional.jpg', { type: 'image/jpeg' }));

    const statusDiv = document.getElementById('register-status');
    statusDiv.innerHTML = `<span style="color:#666">Adding photo to "${itemName}"...</span>`;

    const addRes = await fetch(`/api/items/${itemId}/add-photo`, { method: 'POST', body: form });
    const data = await addRes.json();

    if (addRes.ok) {
        statusDiv.innerHTML = `<span class="success">Added photo to "${data.name}" (${data.photo_count} total)</span>`;
        loadItems();
    } else {
        statusDiv.innerHTML = `<span class="error">${data.detail || 'Failed to add photo'}</span>`;
    }
}

// --- Items List ---

async function loadItems() {
    const res = await fetch('/api/items/');
    const items = await res.json();
    const container = document.getElementById('items-list');

    if (!items.length) {
        container.innerHTML = '<p style="color:#666">No items registered yet. Register an item above to start tracking.</p>';
        return;
    }

    container.innerHTML = items.map(item => {
        const imgTag = item.image_path && item.image_path !== 'text_only'
            ? `<img src="/data/registered/${item.image_path}" alt="${item.name}">`
            : `<div class="item-card-placeholder">Text</div>`;
        const countLabel = item.photo_count > 0
            ? `${item.photo_count} photo${item.photo_count !== 1 ? 's' : ''}`
            : 'text only';
        return `
        <div class="item-card">
            ${imgTag}
            <div class="item-card-body">
                <div class="item-card-info">
                    <span>${item.name}</span>
                    <span class="photo-count">${countLabel}</span>
                </div>
                <div class="item-card-actions">
                    <button class="btn btn-snap btn-small" onclick="addPhotoToItem(${item.id}, '${item.name}')">+ Photo</button>
                    <button class="btn btn-danger btn-small" onclick="deleteItem(${item.id}, '${item.name}')">Remove</button>
                </div>
            </div>
        </div>
    `}).join('');
}

async function deleteItem(id, name) {
    if (!confirm(`Remove "${name}" from tracked items?`)) return;
    const res = await fetch(`/api/items/${id}`, { method: 'DELETE' });
    if (res.ok) {
        loadItems();
    }
}

// --- Init ---

async function init() {
    const [camRes, detRes] = await Promise.all([
        fetch('/api/camera/status'),
        fetch('/api/detection/status')
    ]);
    const camData = await camRes.json();
    const detData = await detRes.json();

    cameraRunning = camData.running;
    detectionRunning = detData.running;

    const camBtn = document.getElementById('btn-camera');
    const detBtn = document.getElementById('btn-detect');

    await loadCameras();

    if (cameraRunning) {
        camBtn.textContent = 'Stop Camera';
        camBtn.classList.add('active');
        detBtn.disabled = false;
        document.getElementById('camera-select').disabled = true;
        document.getElementById('btn-snap').disabled = false;
    }
    if (detectionRunning) {
        detBtn.textContent = 'Stop Detection';
        detBtn.classList.add('active');
    }

    updateStatus();
    loadItems();
}

init();
