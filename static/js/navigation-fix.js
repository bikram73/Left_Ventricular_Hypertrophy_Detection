
    // Navigation Fix JavaScript
    document.addEventListener('DOMContentLoaded', function() {
        // Ensure Bootstrap navbar toggle works
        const navbarToggler = document.querySelector('.navbar-toggler');
        const navbarCollapse = document.querySelector('.navbar-collapse');
        
        if (navbarToggler && navbarCollapse) {
            navbarToggler.addEventListener('click', function() {
                navbarCollapse.classList.toggle('show');
            });
        }
        
        // Ensure all navigation links work
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                const href = this.getAttribute('href');
                if (href && href.startsWith('/')) {
                    // Let the browser handle the navigation normally
                    return true;
                }
            });
        });
        
        // Close mobile menu when clicking outside
        document.addEventListener('click', function(e) {
            const navbar = document.querySelector('.navbar-collapse');
            const toggler = document.querySelector('.navbar-toggler');
            
            if (navbar && navbar.classList.contains('show')) {
                if (!navbar.contains(e.target) && !toggler.contains(e.target)) {
                    navbar.classList.remove('show');
                }
            }
        });
    });
    