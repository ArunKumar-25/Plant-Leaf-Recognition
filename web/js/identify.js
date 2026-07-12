import { Client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

(function () {
    "use strict";

    // Backend is a Hugging Face Space (gradio_app.py) running the same
    // plantify.inference logic api/main.py does -- swapped in while that
    // FastAPI host doesn't have anywhere reliable to run. See DEPLOY.md.
    const GRADIO_BASE = window.PLANTIFY_GRADIO_BASE || "http://localhost:7860";
    let gradioClientPromise = null;

    function getGradioClient() {
        if (!gradioClientPromise) {
            gradioClientPromise = Client.connect(GRADIO_BASE);
        }
        return gradioClientPromise;
    }

    const fileInput = document.getElementById("leaf-file");
    const fileNameEl = document.getElementById("leaf-file-name");
    const preview = document.getElementById("preview");
    const identifyBtn = document.getElementById("identify-btn");

    const cards = {
        ok: document.getElementById("result-ok"),
        uncertain: document.getElementById("result-uncertain"),
        unknown: document.getElementById("result-unknown"),
        error: document.getElementById("result-error"),
    };
    const placeholder = document.getElementById("result-placeholder");
    const qualityWarning = document.getElementById("quality-warning");
    const progress = document.getElementById("result-progress");
    const progressFill = document.getElementById("progress-fill");
    const progressLabel = document.getElementById("progress-label");

    function hideAllCards() {
        Object.values(cards).forEach(function (el) { el.style.display = "none"; });
        if (placeholder) placeholder.style.display = "none";
        if (qualityWarning) qualityWarning.style.display = "none";
        if (progress) progress.style.display = "none";
    }

    // Plain-language stages a visitor can follow, not a literal readout of
    // backend steps -- the API is a single request, so this is a
    // best-guess march through what's actually happening, timed to feel
    // like real progress without ever claiming to reach 100% before the
    // response is back (last stage just sits and waits for it).
    const PROGRESS_STAGES = [
        { label: "Uploading your photo…", pct: 12 },
        { label: "Checking the photo quality…", pct: 32 },
        { label: "Analyzing the leaf shape and texture…", pct: 58 },
        { label: "Comparing against known species…", pct: 82 },
        { label: "Almost done…", pct: 92 },
    ];
    let progressTimer = null;

    function startProgress() {
        if (!progress) return;
        hideAllCards();
        progress.style.display = "block";
        let step = 0;
        progressFill.style.width = PROGRESS_STAGES[0].pct + "%";
        progressLabel.textContent = PROGRESS_STAGES[0].label;
        progressTimer = window.setInterval(function () {
            step = Math.min(step + 1, PROGRESS_STAGES.length - 1);
            progressFill.style.width = PROGRESS_STAGES[step].pct + "%";
            progressLabel.textContent = PROGRESS_STAGES[step].label;
        }, 900);
    }

    function stopProgress() {
        if (progressTimer !== null) {
            window.clearInterval(progressTimer);
            progressTimer = null;
        }
        if (progress) progress.style.display = "none";
    }

    function renderTopK(listEl, topK) {
        listEl.innerHTML = "";
        topK.forEach(function (entry) {
            const pct = Math.round(entry.confidence * 100);
            const li = document.createElement("li");
            li.className = "top-k-row";

            const label = document.createElement("div");
            label.className = "top-k-label";
            const species = document.createElement("span");
            species.textContent = entry.species;
            const pctEl = document.createElement("span");
            pctEl.className = "top-k-pct";
            pctEl.textContent = pct + "%";
            label.appendChild(species);
            label.appendChild(pctEl);

            const bar = document.createElement("div");
            bar.className = "top-k-bar";
            const fill = document.createElement("span");
            fill.style.width = pct + "%";
            bar.appendChild(fill);

            li.appendChild(label);
            li.appendChild(bar);
            listEl.appendChild(li);
        });
    }

    function renderResult(payload) {
        hideAllCards();
        const pct = Math.round(payload.confidence * 100) + "%";

        if (payload.quality_warning && qualityWarning) {
            document.getElementById("quality-warning-text").textContent = payload.quality_warning;
            qualityWarning.style.display = "block";
        }

        if (payload.decision === "ok") {
            document.getElementById("ok-species").textContent = payload.species;
            document.getElementById("ok-confidence").textContent = pct;
            renderTopK(document.getElementById("ok-topk"), payload.top_k.slice(1));
            cards.ok.style.display = "block";
        } else if (payload.decision === "uncertain") {
            document.getElementById("uncertain-species").textContent = payload.species;
            document.getElementById("uncertain-confidence").textContent = pct;
            renderTopK(document.getElementById("uncertain-topk"), payload.top_k);
            renderPlantnetNote(payload.plantnet, "uncertain");
            cards.uncertain.style.display = "block";
        } else {
            renderPlantnetNote(payload.plantnet, "unknown");
            cards.unknown.style.display = "block";
        }
    }

    function renderPlantnetNote(plantnet, prefix) {
        const note = document.getElementById(prefix + "-plantnet");
        if (!note) return;
        if (!plantnet || !plantnet.name) {
            note.style.display = "none";
            return;
        }
        document.getElementById(prefix + "-plantnet-species").textContent = plantnet.name;
        document.getElementById(prefix + "-plantnet-common").textContent = plantnet.common ? "(" + plantnet.common + ")" : "";
        document.getElementById(prefix + "-plantnet-score").textContent = Math.round(plantnet.score * 100) + "%";
        document.getElementById(prefix + "-plantnet-staged-note").textContent = plantnet.staged
            ? "That's confident enough — it's been queued for review before it could ever affect the model."
            : "That's not confident enough to queue for review on its own — try a clearer or more typical photo of this leaf.";
        note.style.display = "block";
    }

    function renderError(message) {
        hideAllCards();
        document.getElementById("error-message").textContent = message;
        cards.error.style.display = "block";
    }

    const MAX_UPLOAD_BYTES = 8 * 1024 * 1024;
    const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"];

    async function predictLeaf(file) {
        // The backend no longer fronts these with its own HTTP status codes
        // (Gradio's client doesn't map errors that granularly), so the
        // same checks the API used to enforce (413/415) happen here first.
        if (file.size > MAX_UPLOAD_BYTES) {
            throw new Error("That file is too large. Try a smaller photo.");
        }
        if (ALLOWED_TYPES.indexOf(file.type) === -1) {
            throw new Error("Unsupported file type. Use JPG, PNG, BMP, TIFF, or WebP.");
        }

        let result;
        try {
            const client = await getGradioClient();
            const response = await client.predict("/identify", [handle_file(file)]);
            result = response.data[0];
        } catch (err) {
            throw new Error("Couldn't reach the prediction service. Is it running?");
        }

        if (result.quality === "reject") {
            throw new Error(
                "This doesn't look like a leaf photo. Try a clear photo of a single leaf on a plain background."
            );
        }
        return result;
    }

    const dropzone = document.getElementById("dropzone");
    const btnLabel = document.getElementById("identify-btn-label");

    function applyFile(file) {
        hideAllCards();
        fileNameEl.textContent = file ? file.name : "No file selected";
        if (!file) {
            preview.style.display = "none";
            identifyBtn.disabled = true;
            if (dropzone) dropzone.classList.remove("has-file");
            return;
        }
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
        identifyBtn.disabled = false;
        if (dropzone) dropzone.classList.add("has-file");
    }

    fileInput.addEventListener("change", function () {
        applyFile(fileInput.files[0]);
    });

    if (dropzone) {
        ["dragenter", "dragover"].forEach(function (evt) {
            dropzone.addEventListener(evt, function (e) {
                e.preventDefault();
                dropzone.classList.add("drag-active");
            });
        });
        ["dragleave", "dragend"].forEach(function (evt) {
            dropzone.addEventListener(evt, function () {
                dropzone.classList.remove("drag-active");
            });
        });
        dropzone.addEventListener("drop", function (e) {
            e.preventDefault();
            dropzone.classList.remove("drag-active");
            const file = e.dataTransfer.files && e.dataTransfer.files[0];
            if (!file) return;
            fileInput.files = e.dataTransfer.files;
            applyFile(file);
        });
    }

    identifyBtn.addEventListener("click", async function () {
        const file = fileInput.files[0];
        if (!file) return;

        identifyBtn.disabled = true;
        identifyBtn.classList.add("is-loading");
        if (btnLabel) btnLabel.textContent = "Identifying...";
        startProgress();

        try {
            const payload = await predictLeaf(file);
            stopProgress();
            renderResult(payload);
        } catch (err) {
            stopProgress();
            renderError(err.message || "Couldn't reach the prediction service. Is the API running?");
        } finally {
            identifyBtn.disabled = false;
            identifyBtn.classList.remove("is-loading");
            if (btnLabel) btnLabel.textContent = "Identify";
        }
    });
})();
