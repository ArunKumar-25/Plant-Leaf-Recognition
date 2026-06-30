(function ($) {
    'use strict';

    function markActiveNav() {
        var current = document.body.getAttribute('data-page');
        if (!current) return;
        var link = document.querySelector('[data-nav="' + current + '"]');
        if (link) link.classList.add('active');
    }

    function afterHeaderLoaded() {
        markActiveNav();

        // Preloader and nav widgets live inside the header partial, so their
        // init has to run after injection, not at initial script load.
        $('.preloader').fadeOut('slow', function () {
            $(this).remove();
        });
        if ($.fn.classyNav) {
            $('#alazeaNav').classyNav();
        }
        if ($.fn.sticky) {
            $('.alazea-main-menu').sticky({ topSpacing: 0 });
        }
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
                console.error('Failed to load partial: ' + url, err);
            });
    }

    document.addEventListener('DOMContentLoaded', function () {
        inject('#partial-header', 'partials/header.html', afterHeaderLoaded);
        inject('#partial-footer', 'partials/footer.html');
    });

})(jQuery);
