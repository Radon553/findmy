let cameraRunning = false;
let detectionRunning = false;

// --- Camera ---

async function toggleCamera() {
    const btn = document.getElementById('btn-camera');
    const detectBtn = document.getElementById('btn-detect');

    if (!cameraRunning) {
        btn.textContent = 'Starting...';
        btn.disabled = true;
        const res = await fetch('/api/camera/start', { method: 'POST' });
        if (res.ok) {
            cameraRunning = true;
            btn.textContent = 'Stop Camera';
            btn.classList.add('active');
            btn.disabled = false;
            detectBtn.disabled = false;
            showFeed();
            updateStatus();
        } else {
            const data = await res.json();
            alert(data.detail || 'Failed to start camera');
            btn.textContent = 'Start Camera';
            btn.disabled = false;
        }
    } else {
        // Stop detection first if running
        if (detectionRunning) {
            await toggleDetection();
        }
        await fetch('/api/camera/stop', { method: 'POST' });
        cameraRunning = false;
        btn.textContent = 'Start Camera';
        btn.classList.remove('active');
        detectBtn.disabled = true;
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

// --- Register ---

async function registerItem(e) {
    e.preventDefault();
    const nameInput = document.getElementById('item-name');
    const photoInput = document.getElementById('item-photo');
    const statusDiv = document.getElementById('register-status');

    const name = nameInput.value.trim();
    if (!name || !photoInput.files.length) return;

    statusDiv.innerHTML = '<span style="color:#666">Registering... (generating CLIP embedding)</span>';

    const form = new FormData();
    form.append('name', name);
    form.append('photo', photoInput.files[0]);

    const res = await fetch('/api/items/register', { method: 'POST', body: form });
    const data = await res.json();

    if (res.ok) {
        statusDiv.innerHTML = `<span class="success">Registered "${data.name}" successfully!</span>`;
        nameInput.value = '';
        photoInput.value = '';
        loadItems();
    } else {
        statusDiv.innerHTML = `<span class="error">${data.detail || 'Registration failed'}</span>`;
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

    container.innerHTML = items.map(item => `
        <div class="item-card">
            <img src="/data/registered/${item.image_path}" alt="${item.name}">
            <div class="item-card-body">
                <span>${item.name}</span>
                <button class="btn btn-danger" onclick="deleteItem(${item.id}, '${item.name}')">Remove</button>
            </div>
        </div>
    `).join('');
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
    // Sync state with server
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

    if (cameraRunning) {
        camBtn.textContent = 'Stop Camera';
        camBtn.classList.add('active');
        detBtn.disabled = false;
        showFeed();
    }
    if (detectionRunning) {
        detBtn.textContent = 'Stop Detection';
        detBtn.classList.add('active');
    }

    updateStatus();
    loadItems();
}

init();
