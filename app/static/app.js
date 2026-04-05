/* === State === */
let camOn = false, detOn = false, pollTimer = null;

/* === Tabs === */
function openTab(t) {
    document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.t === t));
    document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === 'p-' + t));
    if (t === 'feed') loadSightings();
    if (t === 'items') loadItems();
}

/* === Status pill === */
function syncPill() {
    const el = document.getElementById('status-pill');
    if (detOn) { el.textContent = 'Detecting'; el.className = 'pill pill-detecting'; }
    else if (camOn) { el.textContent = 'Camera On'; el.className = 'pill pill-camera'; }
    else { el.textContent = 'Idle'; el.className = 'pill pill-idle'; }
}

/* === Camera === */
async function toggleCamera() {
    const btn = document.getElementById('btn-cam');
    if (!camOn) {
        btn.textContent = 'Starting...'; btn.disabled = true;
        const r = await fetch('/api/camera/start', { method: 'POST' });
        if (r.ok) {
            camOn = true; btn.textContent = 'Stop Camera'; btn.classList.add('active'); btn.disabled = false;
            document.getElementById('btn-det').disabled = false;
            const wrap = document.getElementById('cam-wrap'); wrap.style.display = '';
            document.getElementById('cam-feed').src = '/api/camera/feed?' + Date.now();
        } else {
            const d = await r.json(); alert(d.detail || 'Camera failed');
            btn.textContent = 'Start Camera'; btn.disabled = false;
        }
    } else {
        if (detOn) await toggleDetection();
        await fetch('/api/camera/stop', { method: 'POST' });
        camOn = false; btn.textContent = 'Start Camera'; btn.classList.remove('active');
        document.getElementById('btn-det').disabled = true;
        document.getElementById('cam-wrap').style.display = 'none';
        document.getElementById('cam-feed').src = '';
    }
    syncPill();
}

/* === Detection === */
async function toggleDetection() {
    const btn = document.getElementById('btn-det');
    if (!detOn) {
        btn.textContent = 'Starting...'; btn.disabled = true;
        const r = await fetch('/api/detection/start', { method: 'POST' });
        if (r.ok) { detOn = true; btn.textContent = 'Stop Detection'; btn.classList.add('active'); btn.disabled = false; startSightingPoll(); }
        else { const d = await r.json(); alert(d.detail || 'Detection failed'); btn.textContent = 'Start Detection'; btn.disabled = false; }
    } else {
        await fetch('/api/detection/stop', { method: 'POST' });
        detOn = false; btn.textContent = 'Start Detection'; btn.classList.remove('active'); stopSightingPoll();
    }
    syncPill();
}

/* Auto-refresh sightings while detecting */
function startSightingPoll() { stopSightingPoll(); pollTimer = setInterval(loadSightings, 4000); }
function stopSightingPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

/* === Search === */
async function doSearch(e) {
    e.preventDefault();
    const q = document.getElementById('q').value.trim();
    if (!q) return;
    const out = document.getElementById('search-results');
    out.innerHTML = '<p class="no-results">Searching...</p>';
    const r = await fetch('/api/search/?q=' + encodeURIComponent(q));
    const data = await r.json();
    if (!data.length) { out.innerHTML = '<p class="no-results">No sightings found yet. Point the camera at objects and start detection — items are logged automatically.</p>'; return; }
    out.innerHTML = data.map(s => `
        <div class="result-card">
            <img src="${s.image_url}" alt="${s.item_name}">
            <div class="result-body">
                <h3>${s.item_name}</h3>
                <p class="meta">Last seen <b>${fmtTime(s.last_seen)}</b></p>
                <p class="meta">First seen ${fmtTime(s.first_seen)} &middot; ${s.sighting_count} total sighting${s.sighting_count !== 1 ? 's' : ''}</p>
                <div class="sight-tags">
                    <span class="tag tag-zone">${s.zone}</span>
                    <span class="tag tag-conf">${(s.similarity * 100).toFixed(1)}%</span>
                    ${s.nearby_objects.map(n => `<span class="tag tag-nearby">near ${n}</span>`).join('')}
                </div>
            </div>
        </div>
    `).join('');
}

/* === Sightings feed === */
async function loadSightings() {
    const r = await fetch('/api/search/recent?limit=30');
    const data = await r.json();
    const el = document.getElementById('sightings');
    const countEl = document.getElementById('sight-count');
    const clearBtn = document.getElementById('btn-clear');
    if (!data.length) {
        el.innerHTML = '<div class="empty">No sightings yet &mdash; start camera + detection to begin auto-logging</div>';
        countEl.textContent = '';
        clearBtn.style.display = 'none';
        return;
    }
    countEl.textContent = `${data.length} sighting${data.length !== 1 ? 's' : ''}`;
    clearBtn.style.display = '';
    el.innerHTML = data.map(s => {
        const isAuto = s.source === 'auto';
        const confLabel = isAuto ? `${(s.similarity * 100).toFixed(0)}% conf` : `${(s.similarity * 100).toFixed(1)}% match`;
        return `
        <div class="sight-card${isAuto ? ' sight-auto' : ''}">
            <img src="/data/images/${s.image_path}" alt="${s.item_name}" loading="lazy">
            <div class="sight-info">
                <div class="sight-header">
                    <span class="name">${s.item_name}${isAuto ? ' <span class="badge-auto">auto</span>' : ''}</span>
                    <button class="btn-delete" onclick="deleteSighting(${s.id})" title="Delete sighting">&times;</button>
                </div>
                <span class="detail">${fmtTime(s.timestamp)}</span>
                <div class="sight-tags">
                    <span class="tag tag-zone">${s.zone || 'center'}</span>
                    <span class="tag tag-conf">${confLabel}</span>
                    <span class="tag tag-source">${s.source || 'camera'}</span>
                    ${(s.nearby_objects || []).map(n => `<span class="tag tag-nearby">near ${n}</span>`).join('')}
                </div>
            </div>
        </div>`;
    }).join('');
}

/* === Delete sightings === */
async function deleteSighting(id) {
    await fetch('/api/search/sightings/' + id, { method: 'DELETE' });
    loadSightings();
}
async function clearAllSightings() {
    if (!confirm('Delete all sightings? This cannot be undone.')) return;
    await fetch('/api/search/sightings', { method: 'DELETE' });
    loadSightings();
}

/* === Video upload === */
function onDragOver(e) { e.preventDefault(); document.getElementById('drop').classList.add('drag-over'); }
function onDragLeave(e) { document.getElementById('drop').classList.remove('drag-over'); }
function onDrop(e) {
    e.preventDefault(); document.getElementById('drop').classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('video/')) uploadVideo(f);
}
function onFileSelect(e) { const f = e.target.files[0]; if (f) uploadVideo(f); }

async function uploadVideo(file) {
    const wrap = document.getElementById('prog-wrap'); wrap.style.display = '';
    const label = document.getElementById('prog-label');
    const pct = document.getElementById('prog-pct');
    const bar = document.getElementById('prog-bar');
    const detail = document.getElementById('prog-detail');
    label.textContent = 'Uploading...'; pct.textContent = ''; bar.style.width = '0%'; detail.textContent = '';

    const fd = new FormData(); fd.append('file', file);
    const r = await fetch('/api/video/upload', { method: 'POST', body: fd });
    if (!r.ok) { const d = await r.json(); label.textContent = 'Error: ' + (d.detail || 'Upload failed'); return; }
    const job = await r.json();
    pollVideoJob(job.job_id);
}

function pollVideoJob(jobId) {
    const iv = setInterval(async () => {
        const r = await fetch('/api/video/status/' + jobId);
        const j = await r.json();
        const pct = document.getElementById('prog-pct');
        const bar = document.getElementById('prog-bar');
        const label = document.getElementById('prog-label');
        const detail = document.getElementById('prog-detail');
        const p = Math.round(j.progress * 100);
        pct.textContent = p + '%'; bar.style.width = p + '%';
        detail.textContent = `${j.processed_frames}/${j.total_frames} frames \u00b7 ${j.sightings_found} sighting${j.sightings_found !== 1 ? 's' : ''} found`;
        if (j.status === 'completed') {
            clearInterval(iv); label.textContent = 'Done!'; loadSightings(); openTab('feed');
        } else if (j.status === 'error') {
            clearInterval(iv); label.textContent = 'Error: ' + (j.error || 'Processing failed');
        } else { label.textContent = 'Processing...'; }
    }, 1000);
}

/* === Register item === */
async function registerItem(e) {
    e.preventDefault();
    const nameEl = document.getElementById('item-name');
    const photoEl = document.getElementById('item-photo');
    const status = document.getElementById('reg-status');
    const name = nameEl.value.trim();
    if (!name || !photoEl.files.length) { status.innerHTML = '<span class="error">Name and at least one photo required</span>'; return; }
    status.innerHTML = '<span style="color:var(--muted)">Registering... (generating embeddings)</span>';
    const fd = new FormData(); fd.append('name', name);
    for (const f of photoEl.files) fd.append('photos', f);
    const r = await fetch('/api/items/register', { method: 'POST', body: fd });
    const d = await r.json();
    if (r.ok) { status.innerHTML = `<span class="success">Registered "${d.name}" with ${d.photo_count} photo(s)</span>`; nameEl.value = ''; photoEl.value = ''; loadItems(); }
    else { status.innerHTML = `<span class="error">${typeof d.detail === 'string' ? d.detail : 'Registration failed'}</span>`; }
}

/* === Items list === */
async function loadItems() {
    const r = await fetch('/api/items/');
    const items = await r.json();
    const el = document.getElementById('items-grid');
    if (!items.length) { el.innerHTML = '<div class="empty">No items registered yet</div>'; return; }
    el.innerHTML = items.map(i => `
        <div class="item-card">
            <img src="/data/registered/${i.image_path}" alt="${i.name}">
            <div class="item-body">
                <div class="item-top">
                    <span>${i.name}</span>
                    <span class="photo-cnt">${i.photo_count} photo${i.photo_count !== 1 ? 's' : ''}</span>
                </div>
                <button class="btn btn-red" onclick="deleteItem(${i.id},'${i.name}')">Remove</button>
            </div>
        </div>
    `).join('');
}
async function deleteItem(id, name) {
    if (!confirm('Remove "' + name + '"?')) return;
    await fetch('/api/items/' + id, { method: 'DELETE' }); loadItems();
}

/* === Helpers === */
function fmtTime(ts) { try { return new Date(ts + 'Z').toLocaleString(); } catch { return ts; } }

/* === Init === */
(async () => {
    const [cr, dr] = await Promise.all([fetch('/api/camera/status'), fetch('/api/detection/status')]);
    const cd = await cr.json(), dd = await dr.json();
    camOn = cd.running; detOn = dd.running;
    if (camOn) { document.getElementById('btn-cam').textContent = 'Stop Camera'; document.getElementById('btn-cam').classList.add('active'); document.getElementById('btn-det').disabled = false; document.getElementById('cam-wrap').style.display = ''; document.getElementById('cam-feed').src = '/api/camera/feed?' + Date.now(); }
    if (detOn) { document.getElementById('btn-det').textContent = 'Stop Detection'; document.getElementById('btn-det').classList.add('active'); startSightingPoll(); }
    syncPill(); loadSightings(); loadItems();
})();
