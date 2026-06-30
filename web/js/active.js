(function ($) {
    'use strict';

    var browserWindow = $(window);

    // Note: preloader fade-out, classyNav init, and the sticky nav init all
    // moved to js/partials.js, since that markup lives in the header partial
    // and doesn't exist in the DOM until partials.js injects it.

    // :: 3.0 Search Active Code
    $('#searchIcon').on('click', function () {
        $('.search-form').toggleClass('active');
    });
    $('.closeIcon').on('click', function () {
        $('.search-form').removeClass('active');
    });

    // :: 4.0 Hero Slider Active Code
    if ($.fn.owlCarousel) {
        $('.hero-post-slides').owlCarousel({
            items: 1,
            margin: 0,
            loop: true,
            nav: false,
            dots: false,
            autoplay: true,
            center: true,
            autoplayTimeout: 5000,
            smartSpeed: 1000
        });
    }

    // :: 5.0 ScrollUp Active Code
    if ($.fn.scrollUp) {
        browserWindow.scrollUp({
            scrollSpeed: 1500,
            scrollText: '<i class="fa fa-angle-up"></i>'
        });
    }

    // :: 7.0 Tooltip Active Code
    if ($.fn.tooltip) {
        $('[data-toggle="tooltip"]').tooltip()
    }

    // :: 8.0 Prevent default "#" link clicks
    $('a[href="#"]').on('click', function ($) {
        $.preventDefault();
    });

    // :: 9.0 WOW (scroll fade-in) Active Code
    if (browserWindow.width() > 767) {
        new WOW().init();
    }

})(jQuery);
