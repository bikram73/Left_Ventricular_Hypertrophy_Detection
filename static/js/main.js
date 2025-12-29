/**
 * Advanced JavaScript for LVH Detection System
 * Handles file uploads, form validation, animations, and UI interactions
 */

class LVHDetectionApp {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeAnimations();
        this.setupFormValidation();
        this.initializeTooltips();
    }

    setupEventListeners() {
        // File upload handling
        document.querySelectorAll('input[type="file"]').forEach(input => {
            input.addEventListener('change', this.handleFileUpload.bind(this));
        });

        // Form submission
        const predictForm = document.querySelector('form[action="/predict"]');
        if (predictForm) {
            predictForm.addEventListener('submit', this.handleFormSubmission.bind(this));
        }

        // Navigation smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', this.handleSmoothScroll.bind(this));
        });

        // Responsive navbar
        this.setupResponsiveNavbar();

        // Real-time form validation
        this.setupRealTimeValidation();
    }

    handleFileUpload(event) {
        const input = event.target;
        const file = input.files[0];
        const section = input.closest('.upload-section');
        
        if (file) {
            this.showFileUploadFeedback(section, file);
            this.validateFileType(file, input);
        } else {
            this.clearFileUploadFeedback(section);
        }
    }

    showFileUploadFeedback(section, file) {
        section.style.borderColor = '#28a745';
        section.style.backgroundColor = 'rgba(40, 167, 69, 0.1)';
        section.classList.add('file-uploaded');
        
        // Remove existing feedback
        const existingFeedback = section.querySelector('.file-feedback');
        if (existingFeedback) {
            existingFeedback.remove();
        }
        
        // Add new feedback
        const feedback = document.createElement('div');
        feedback.className = 'file-feedback text-success mt-2';
        feedback.innerHTML = `
            <i class="fas fa-check-circle"></i> 
            <strong>${file.name}</strong>
            <br>
            <small class="text-muted">Size: ${this.formatFileSize(file.size)} | Type: ${file.type || 'Unknown'}</small>
        `;
        
        section.appendChild(feedback);

        // Add animation
        feedback.style.opacity = '0';
        feedback.style.transform = 'translateY(10px)';
        
        setTimeout(() => {
            feedback.style.transition = 'all 0.3s ease';
            feedback.style.opacity = '1';
            feedback.style.transform = 'translateY(0)';
        }, 100);
    }

    clearFileUploadFeedback(section) {
        section.style.borderColor = '#667eea';
        section.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
        section.classList.remove('file-uploaded');
        
        const feedback = section.querySelector('.file-feedback');
        if (feedback) {
            feedback.remove();
        }
    }

    validateFileType(file, input) {
        const allowedTypes = {
            'ecg_file': ['.csv', '.txt'],
            'mri_file': ['.png', '.jpg', '.jpeg', '.dcm'],
            'ct_file': ['.png', '.jpg', '.jpeg', '.dcm']
        };

        const inputName = input.name;
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (allowedTypes[inputName] && !allowedTypes[inputName].includes(fileExtension)) {
            this.showAlert(`Invalid file type for ${inputName.replace('_', ' ').toUpperCase()}. Allowed types: ${allowedTypes[inputName].join(', ')}`, 'warning');
            input.value = '';
            this.clearFileUploadFeedback(input.closest('.upload-section'));
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    handleFormSubmission(event) {
        const form = event.target;
        const submitButton = form.querySelector('button[type="submit"]');
        
        if (!this.validateForm(form)) {
            event.preventDefault();
            return;
        }

        // Show loading state
        if (submitButton) {
            const originalText = submitButton.innerHTML;
            submitButton.innerHTML = '<span class="spinner-custom"></span> Analyzing...';
            submitButton.disabled = true;
            
            // Re-enable after 30 seconds as fallback
            setTimeout(() => {
                submitButton.innerHTML = originalText;
                submitButton.disabled = false;
            }, 30000);
        }

        this.showAlert('Processing your data... Please wait.', 'info');
    }

    validateForm(form) {
        let isValid = true;
        const requiredFields = form.querySelectorAll('[required]');
        
        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                this.showFieldError(field, 'This field is required');
                isValid = false;
            } else {
                this.clearFieldError(field);
            }
        });

        // Clinical data validation
        const age = form.querySelector('#age');
        if (age && age.value) {
            const ageValue = parseInt(age.value);
            if (ageValue < 0 || ageValue > 120) {
                this.showFieldError(age, 'Age must be between 0 and 120');
                isValid = false;
            }
        }

        const restingBp = form.querySelector('#resting_bp');
        if (restingBp && restingBp.value) {
            const bpValue = parseInt(restingBp.value);
            if (bpValue < 50 || bpValue > 300) {
                this.showFieldError(restingBp, 'Blood pressure must be between 50 and 300 mmHg');
                isValid = false;
            }
        }

        const cholesterol = form.querySelector('#cholesterol');
        if (cholesterol && cholesterol.value) {
            const cholValue = parseInt(cholesterol.value);
            if (cholValue < 100 || cholValue > 600) {
                this.showFieldError(cholesterol, 'Cholesterol must be between 100 and 600 mg/dl');
                isValid = false;
            }
        }

        return isValid;
    }

    showFieldError(field, message) {
        field.classList.add('is-invalid');
        
        // Remove existing error
        const existingError = field.parentNode.querySelector('.invalid-feedback');
        if (existingError) {
            existingError.remove();
        }
        
        // Add new error
        const errorDiv = document.createElement('div');
        errorDiv.className = 'invalid-feedback';
        errorDiv.textContent = message;
        field.parentNode.appendChild(errorDiv);
    }

    clearFieldError(field) {
        field.classList.remove('is-invalid');
        const errorDiv = field.parentNode.querySelector('.invalid-feedback');
        if (errorDiv) {
            errorDiv.remove();
        }
    }

    setupRealTimeValidation() {
        const inputs = document.querySelectorAll('input[type="number"], select');
        
        inputs.forEach(input => {
            input.addEventListener('blur', () => {
                const form = input.closest('form');
                if (form) {
                    this.validateForm(form);
                }
            });

            input.addEventListener('input', () => {
                if (input.classList.contains('is-invalid')) {
                    this.clearFieldError(input);
                }
            });
        });
    }

    handleSmoothScroll(event) {
        event.preventDefault();
        const targetId = event.currentTarget.getAttribute('href');
        const targetSection = document.querySelector(targetId);
        
        if (targetSection) {
            const offsetTop = targetSection.offsetTop - 80; // Account for navbar
            window.scrollTo({
                top: offsetTop,
                behavior: 'smooth'
            });
        }
    }

    setupResponsiveNavbar() {
        const navbar = document.querySelector('.navbar');
        const navbarToggler = document.querySelector('.navbar-toggler');
        const navbarCollapse = document.querySelector('.navbar-collapse');
        
        // Close navbar when clicking outside
        document.addEventListener('click', (event) => {
            if (navbarCollapse && navbarCollapse.classList.contains('show')) {
                if (!navbar.contains(event.target)) {
                    navbarToggler.click();
                }
            }
        });

        // Add scroll effect to navbar
        window.addEventListener('scroll', () => {
            if (navbar) {
                if (window.scrollY > 50) {
                    navbar.style.backgroundColor = 'rgba(255, 255, 255, 0.98)';
                    navbar.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
                } else {
                    navbar.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
                    navbar.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
                }
            }
        });
    }

    initializeAnimations() {
        // Animate elements on scroll
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -100px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, observerOptions);

        // Observe feature cards
        document.querySelectorAll('.feature-card, .stats-card, .detail-card').forEach(card => {
            observer.observe(card);
        });

        // Animate confidence bars
        this.animateConfidenceBars();
    }

    animateConfidenceBars() {
        const confidenceBars = document.querySelectorAll('.confidence-fill');
        
        confidenceBars.forEach(bar => {
            const targetWidth = bar.style.width;
            bar.style.width = '0%';
            
            setTimeout(() => {
                bar.style.transition = 'width 2s ease-in-out';
                bar.style.width = targetWidth;
            }, 500);
        });
    }

    initializeTooltips() {
        // Initialize Bootstrap tooltips if available
        if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(tooltipTriggerEl => {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    }

    showAlert(message, type = 'info') {
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show position-fixed" 
                 style="top: 20px; right: 20px; z-index: 9999; min-width: 300px;" role="alert">
                <strong>${this.getAlertIcon(type)}</strong> ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', alertHtml);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            const alerts = document.querySelectorAll('.alert.position-fixed');
            if (alerts.length > 0) {
                alerts[0].remove();
            }
        }, 5000);
    }

    getAlertIcon(type) {
        const icons = {
            'success': '<i class="fas fa-check-circle"></i>',
            'warning': '<i class="fas fa-exclamation-triangle"></i>',
            'danger': '<i class="fas fa-exclamation-circle"></i>',
            'info': '<i class="fas fa-info-circle"></i>'
        };
        return icons[type] || icons['info'];
    }

    // Utility function for API calls
    async makeAPICall(endpoint, data = null, method = 'GET') {
        try {
            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                }
            };

            if (data && method !== 'GET') {
                options.body = JSON.stringify(data);
            }

            const response = await fetch(endpoint, options);
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.message || 'API call failed');
            }

            return result;
        } catch (error) {
            console.error('API Error:', error);
            this.showAlert(`Error: ${error.message}`, 'danger');
            throw error;
        }
    }

    // Health check function
    async checkSystemHealth() {
        try {
            const health = await this.makeAPICall('/health');
            console.log('System Health:', health);
            return health;
        } catch (error) {
            console.error('Health check failed:', error);
            return null;
        }
    }

    // Demo prediction function for testing
    async makeDemoPrediction(clinicalData) {
        try {
            const result = await this.makeAPICall('/api/predict', {
                clinical_data: clinicalData
            }, 'POST');
            
            this.showAlert('Prediction completed successfully!', 'success');
            return result;
        } catch (error) {
            this.showAlert('Prediction failed. Please try again.', 'danger');
            return null;
        }
    }

    // Search functionality for help page
    initializeHelpSearch() {
        const helpSearchInput = document.getElementById('helpSearch');
        const helpSearchResults = document.getElementById('helpSearchResults');
        
        if (!helpSearchInput) return;
        
        helpSearchInput.addEventListener('input', (e) => {
            const query = e.target.value.trim().toLowerCase();
            
            if (query.length < 2) {
                this.clearHelpSearchResults();
                return;
            }
            
            this.searchHelpContent(query);
        });
    }
    
    searchHelpContent(query) {
        const helpSections = document.querySelectorAll('.help-container, .step-card, .faq-section');
        const results = [];
        
        helpSections.forEach(section => {
            const title = section.querySelector('h1, h2, h3, h4, h5, h6')?.textContent || '';
            const content = section.textContent.toLowerCase();
            
            if (content.includes(query)) {
                const snippet = this.extractSnippet(section.textContent, query);
                results.push({
                    title: title,
                    snippet: snippet,
                    element: section
                });
            }
        });
        
        this.displayHelpSearchResults(results, query);
    }
    
    extractSnippet(text, query, maxLength = 150) {
        const index = text.toLowerCase().indexOf(query.toLowerCase());
        if (index === -1) return text.substring(0, maxLength) + '...';
        
        const start = Math.max(0, index - 50);
        const end = Math.min(text.length, index + query.length + 100);
        
        let snippet = text.substring(start, end);
        if (start > 0) snippet = '...' + snippet;
        if (end < text.length) snippet = snippet + '...';
        
        return snippet;
    }
    
    displayHelpSearchResults(results, query) {
        const helpSearchResults = document.getElementById('helpSearchResults');
        if (!helpSearchResults) return;
        
        if (results.length === 0) {
            helpSearchResults.innerHTML = '<p class="text-muted">No results found</p>';
            helpSearchResults.style.display = 'block';
            return;
        }
        
        const resultsHtml = results.map(result => `
            <div class="help-search-result" onclick="this.scrollToSection('${result.element.id}')">
                <h6>${this.highlightText(result.title, query)}</h6>
                <p class="text-muted small">${this.highlightText(result.snippet, query)}</p>
            </div>
        `).join('');
        
        helpSearchResults.innerHTML = resultsHtml;
        helpSearchResults.style.display = 'block';
    }
    
    highlightText(text, query) {
        const regex = new RegExp(`(${query})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }
    
    scrollToSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
    
    clearHelpSearchResults() {
        const helpSearchResults = document.getElementById('helpSearchResults');
        if (helpSearchResults) {
            helpSearchResults.style.display = 'none';
        }
    }

    // Results page filtering
    initializeResultsFiltering() {
        const filterButtons = document.querySelectorAll('.result-filter-btn');
        const resultSections = document.querySelectorAll('.result-section');
        
        if (filterButtons.length === 0) return;
        
        filterButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const filter = e.target.dataset.filter;
                this.filterResults(filter, filterButtons, resultSections);
            });
        });
    }
    
    filterResults(filter, filterButtons, resultSections) {
        // Update active button
        filterButtons.forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-filter="${filter}"]`).classList.add('active');
        
        // Show/hide sections
        resultSections.forEach(section => {
            if (filter === 'all' || section.dataset.category === filter) {
                section.style.display = 'block';
                section.style.animation = 'fadeIn 0.3s ease';
            } else {
                section.style.display = 'none';
            }
        });
    }
}

// Utility functions
function formatDate(date) {
    return new Date(date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

function formatTime(date) {
    return new Date(date).toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        app.showAlert('Copied to clipboard!', 'success');
    }).catch(err => {
        console.error('Failed to copy:', err);
        app.showAlert('Failed to copy to clipboard', 'warning');
    });
}

// CSS for animations
const animationCSS = `
    .feature-card, .stats-card, .detail-card {
        opacity: 0;
        transform: translateY(30px);
        transition: all 0.6s ease;
    }
    
    .animate-in {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
    
    .spinner-custom {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid #f3f3f3;
        border-top: 2px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;

// Add animation CSS to page
if (!document.querySelector('#animation-styles')) {
    const style = document.createElement('style');
    style.id = 'animation-styles';
    style.textContent = animationCSS;
    document.head.appendChild(style);
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new LVHDetectionApp();
    
    // Perform initial health check
    app.checkSystemHealth().then(health => {
        if (health && health.status === 'healthy') {
            console.log('✅ LVH Detection System is healthy');
        } else {
            console.warn('⚠️ System health check failed');
        }
    });
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LVHDetectionApp };
}