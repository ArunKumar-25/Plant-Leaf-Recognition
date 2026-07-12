(function () {
    "use strict";

    var prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // Cross-page anchor links (e.g. "About" -> index.html#about-us) land
    // with an instant browser jump by default, unlike a same-page anchor
    // click which uses the CSS smooth scroll-behavior -- that mismatch is
    // what makes nav feel inconsistent depending on which page you're
    // clicking from. Undo the instant jump here (as early as this script
    // runs) so smoothScrollToHash can animate to it instead, once the page
    // has actually finished laying out.
    if (window.location.hash && !prefersReduced && "scrollRestoration" in history) {
        history.scrollRestoration = "manual";
        window.scrollTo(0, 0);
    }

    function smoothScrollToHash() {
        if (!window.location.hash || prefersReduced) return;
        var target = document.querySelector(window.location.hash);
        if (!target) return;
        target.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function markActiveNav() {
        var links = document.querySelectorAll(".nav-links a");
        links.forEach(function (a) { a.classList.remove("active"); });

        // A hash match (e.g. "#about-us") takes priority over the page-level
        // match -- otherwise a same-page link like "About" (index.html#about-us,
        // data-page="home") could never highlight, since data-page only knows
        // which page you're on, not which section within it.
        var hash = window.location.hash;
        if (hash) {
            var hashLink = document.querySelector('.nav-links a[href$="' + hash + '"]');
            if (hashLink) {
                hashLink.classList.add("active");
                return;
            }
        }

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

    // A same-page anchor click (already on index.html, clicking "About")
    // changes the hash without a reload -- markActiveNav only runs once on
    // initial load otherwise, so the active link would never update.
    window.addEventListener("hashchange", markActiveNav);

    // Wait for `load`, not DOMContentLoaded -- images below the target can
    // still shift layout after DOM parsing finishes, and scrolling before
    // that settles would animate to the wrong position.
    window.addEventListener("load", smoothScrollToHash);
})();
