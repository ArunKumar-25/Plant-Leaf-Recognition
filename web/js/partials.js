(function () {
    "use strict";

    function markActiveNav() {
        var current = document.body.getAttribute("data-page");
        if (!current) return;
        var link = document.querySelector('[data-nav="' + current + '"]');
        if (link) link.classList.add("active");
    }

    function initMobileNav() {
        var toggle = document.getElementById("nav-toggle");
        var links = document.getElementById("nav-links");
        if (!toggle || !links) return;
        toggle.addEventListener("click", function () {
            var open = links.classList.toggle("open");
            toggle.setAttribute("aria-expanded", open ? "true" : "false");
        });
        links.addEventListener("click", function (event) {
            if (event.target.tagName === "A") {
                links.classList.remove("open");
                toggle.setAttribute("aria-expanded", "false");
            }
        });
    }

    function afterHeaderLoaded() {
        markActiveNav();
        initMobileNav();
    }

    function afterFooterLoaded() {
        var yearEl = document.getElementById("footer-year");
        if (yearEl) yearEl.textContent = new Date().getFullYear();
    }

    function inject(selector, url, after) {
        var mount = document.querySelector(selector);
        if (!mount) return;
        fetch(url)
            .then(function (res) { return res.text(); })
            .then(function (html) {
                mount.outerHTML = html;
                if (after) after();
            })
            .catch(function (err) {
                console.error("Failed to load partial: " + url, err);
            });
    }

    document.addEventListener("DOMContentLoaded", function () {
        inject("#partial-header", "partials/header.html", afterHeaderLoaded);
        inject("#partial-footer", "partials/footer.html", afterFooterLoaded);
    });
})();
