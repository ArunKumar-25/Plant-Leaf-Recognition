(function () {
    "use strict";

    const API_BASE = window.PLANTIFY_API_BASE || "http://localhost:8000";

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

    function hideAllCards() {
        Object.values(cards).forEach(function (el) { el.style.display = "none"; });
    }

    function renderTopK(listEl, topK) {
        listEl.innerHTML = "";
        topK.forEach(function (entry) {
            const li = document.createElement("li");
            li.textContent = entry.species + " (" + Math.round(entry.confidence * 100) + "%)";
            listEl.appendChild(li);
        });
    }

    function renderResult(payload) {
        hideAllCards();
        const pct = Math.round(payload.confidence * 100) + "%";

        if (payload.decision === "ok") {
            document.getElementById("ok-species").textContent = payload.species;
            document.getElementById("ok-confidence").textContent = pct;
            renderTopK(document.getElementById("ok-topk"), payload.top_k.slice(1));
            cards.ok.style.display = "block";
        } else if (payload.decision === "uncertain") {
            document.getElementById("uncertain-species").textContent = payload.species;
            document.getElementById("uncertain-confidence").textContent = pct;
            renderTopK(document.getElementById("uncertain-topk"), payload.top_k);
            cards.uncertain.style.display = "block";
        } else {
            cards.unknown.style.display = "block";
        }
    }

    function renderError(message) {
        hideAllCards();
        document.getElementById("error-message").textContent = message;
        cards.error.style.display = "block";
    }

    async function predictLeaf(file) {
        const form = new FormData();
        form.append("file", file);

        const res = await fetch(API_BASE + "/predict", {
            method: "POST",
            body: form,
        });

        if (!res.ok) {
            const detail = await res.json().catch(function () { return {}; });
            if (res.status === 413) {
                throw new Error("That file is too large. Try a smaller photo.");
            }
            if (res.status === 415) {
                throw new Error("Unsupported file type. Use JPG, PNG, BMP, TIFF, or WebP.");
            }
            throw new Error(detail.detail || ("Prediction failed (" + res.status + ")"));
        }
        return res.json();
    }

    fileInput.addEventListener("change", function () {
        hideAllCards();
        const file = fileInput.files[0];
        fileNameEl.textContent = file ? file.name : "No file selected";
        if (!file) {
            preview.style.display = "none";
            identifyBtn.disabled = true;
            return;
        }
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
        identifyBtn.disabled = false;
    });

    identifyBtn.addEventListener("click", async function () {
        const file = fileInput.files[0];
        if (!file) return;

        identifyBtn.disabled = true;
        const originalText = identifyBtn.textContent;
        identifyBtn.textContent = "Identifying...";
        hideAllCards();

        try {
            const payload = await predictLeaf(file);
            renderResult(payload);
        } catch (err) {
            renderError(err.message || "Couldn't reach the prediction service. Is the API running?");
        } finally {
            identifyBtn.disabled = false;
            identifyBtn.textContent = originalText;
        }
    });
})();
