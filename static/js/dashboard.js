/**
 * Dashboard JavaScript for LVH Detection System
 * Handles real-time metrics, analytics, and dashboard interactions
 */

class DashboardManager {
    constructor() {
        this.updateInterval = 30000; // 30 seconds
        this.metricsTimer = null;
        this.predictionsTimer = null;
        this.accuracyChart = null;
        this.startTime = Date.now();
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadInitialData();
        this.startAutoRefresh();
        this.initializeCharts();
    }
    
    setupEventListeners() {
        // Refresh buttons
        const refreshBtn = document.querySelector('[onclick="refreshPredictions()"]');
        if (refreshBtn) {
            refreshBtn.onclick = () => this.loadRecentPredictions();
        }
        
        // Auto-refresh on visibility change
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                this.loadAllData();
            }
        });
        
        // Handle window focus
        window.addEventListener('focus', () => {
            this.loadAllData();
        });
    }
    
    async loadInitialData() {
        try {
            await Promise.all([
                this.loadSystemMetrics(),
                this.loadAnalytics(),
                this.loadRecentPredictions(),
                this.loadModelPerformance()
            ]);
        } catch (error) {
            console.error('Error loading initial dashboard data:', error);
            this.showError('Failed to load dashboard data');
        }
    }
    
    async loadAllData() {
        try {
            await this.loadInitialData();
        } catch (error) {
            console.error('Error refreshing dashboard data:', error);
        }
    }
    
    async loadSystemMetrics() {
        try {
            const response = await fetch('/api/dashboard/metrics');
            const data = await response.json();
            
            if (data.success) {
                this.updateSystemMetrics(data.metrics);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error loading system metrics:', error);
            this.showMetricsError();
        }
    }
    
    updateSystemMetrics(metrics) {
        // Update system status
        const statusElement = document.getElementById('systemStatus');
        const statusText = document.getElementById('statusText');
        const statusIcon = document.getElementById('statusIcon');
        
        if (statusElement && statusText && statusIcon) {
            statusText.textContent = metrics.status;
            
            // Update status styling
            statusElement.className = 'system-status';
            if (metrics.status === 'WARNING') {
                statusElement.classList.add('warning');
                statusIcon.className = 'fas fa-exclamation-triangle';
            } else if (metrics.status === 'CRITICAL') {
                statusElement.classList.add('critical');
                statusIcon.className = 'fas fa-exclamation-circle';
            } else {
                statusIcon.className = 'fas fa-check-circle';
            }
        }
        
        // Update uptime
        const uptimeElement = document.getElementById('uptimeText');
        if (uptimeElement) {
            const uptime = this.calculateUptime();
            uptimeElement.textContent = uptime;
        }
        
        // Update CPU usage
        this.updateMetricCard('cpuUsage', `${metrics.cpu_usage.toFixed(1)}%`);
        this.updateProgressBar('cpuProgress', metrics.cpu_usage);
        
        // Update RAM usage
        const ramPercentage = (metrics.ram_usage / metrics.ram_total) * 100;
        this.updateMetricCard('ramUsage', `${metrics.ram_usage.toFixed(1)} GB`);
        this.updateProgressBar('ramProgress', ramPercentage);
        
        // Update Disk usage
        this.updateMetricCard('diskUsage', `${metrics.disk_usage.toFixed(1)}%`);
        this.updateProgressBar('diskProgress', metrics.disk_usage);
        
        // Update API response time
        this.updateMetricCard('apiResponse', `${metrics.api_response_time.toFixed(0)} ms`);
        
        // Update API status
        const apiStatus = document.getElementById('apiStatus');
        if (apiStatus) {
            if (metrics.api_response_time < 100) {
                apiStatus.innerHTML = '<i class="fas fa-check-circle"></i> Excellent';
                apiStatus.className = 'text-success';
            } else if (metrics.api_response_time < 500) {
                apiStatus.innerHTML = '<i class="fas fa-clock"></i> Good';
                apiStatus.className = 'text-warning';
            } else {
                apiStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Slow';
                apiStatus.className = 'text-danger';
            }
        }
    }
    
    updateMetricCard(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
            element.classList.add('pulse');
            setTimeout(() => element.classList.remove('pulse'), 1000);
        }
    }
    
    updateProgressBar(elementId, percentage) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.width = `${Math.min(percentage, 100)}%`;
            
            // Update color based on percentage
            element.className = 'progress-bar';
            if (percentage > 90) {
                element.classList.add('bg-danger');
            } else if (percentage > 70) {
                element.classList.add('bg-warning');
            } else {
                element.classList.add('bg-success');
            }
        }
    }
    
    async loadAnalytics() {
        try {
            const response = await fetch('/api/dashboard/analytics');
            const data = await response.json();
            
            if (data.success) {
                this.updateAnalytics(data.analytics);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error loading analytics:', error);
            this.showAnalyticsError();
        }
    }
    
    updateAnalytics(analytics) {
        // Update total predictions
        this.updateStatsCard('totalPredictions', analytics.total_predictions);
        this.updateStatsChange('totalChange', '+12%', true); // Simulated change
        
        // Update today's accuracy
        const todayAccuracy = analytics.accuracy_trend[analytics.accuracy_trend.length - 1] || 0;
        this.updateStatsCard('todayAccuracy', `${(todayAccuracy * 100).toFixed(1)}%`);
        this.updateStatsChange('accuracyChange', '+1.2%', true);
        
        // Update week predictions
        this.updateStatsCard('weekPredictions', analytics.week_predictions);
        this.updateStatsChange('weekChange', '+8%', true);
        
        // Update average processing time
        this.updateStatsCard('avgProcessingTime', `${analytics.avg_processing_time.toFixed(1)}s`);
        this.updateStatsChange('processingChange', '0s', null);
        
        // Update result distribution
        this.updateStatsCard('lvhDetected', analytics.lvh_detected);
        this.updateStatsCard('normalResults', analytics.normal_results);
        this.updateStatsCard('lowConfidence', analytics.low_confidence);
        this.updateStatsCard('activeUsers', analytics.active_users);
        
        // Update percentages
        const total = analytics.lvh_detected + analytics.normal_results;
        if (total > 0) {
            const lvhPercent = ((analytics.lvh_detected / total) * 100).toFixed(1);
            const normalPercent = ((analytics.normal_results / total) * 100).toFixed(1);
            const lowConfPercent = ((analytics.low_confidence / total) * 100).toFixed(1);
            
            this.updateElement('lvhPercentage', `${lvhPercent}%`);
            this.updateElement('normalPercentage', `${normalPercent}%`);
            this.updateElement('lowConfPercentage', `${lowConfPercent}%`);
        }
        
        // Update accuracy trend chart
        this.updateAccuracyChart(analytics.accuracy_trend);
    }
    
    updateStatsCard(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            // Animate number change
            this.animateNumber(element, parseInt(element.textContent) || 0, value);
        }
    }
    
    updateStatsChange(elementId, change, isPositive) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `<i class="fas fa-${isPositive ? 'arrow-up' : isPositive === false ? 'arrow-down' : 'minus'}"></i> ${change}`;
            element.className = `stats-change ${isPositive ? 'positive' : isPositive === false ? 'negative' : ''}`;
        }
    }
    
    animateNumber(element, start, end) {
        const duration = 1000;
        const startTime = Date.now();
        const endValue = typeof end === 'string' ? parseFloat(end) : end;
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const current = start + (endValue - start) * progress;
            
            if (typeof end === 'string' && end.includes('%')) {
                element.textContent = `${current.toFixed(1)}%`;
            } else if (typeof end === 'string' && end.includes('s')) {
                element.textContent = `${current.toFixed(1)}s`;
            } else {
                element.textContent = Math.round(current);
            }
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    async loadRecentPredictions() {
        try {
            const response = await fetch('/api/dashboard/recent-predictions?limit=10');
            const data = await response.json();
            
            if (data.success) {
                this.updateRecentPredictions(data.predictions);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error loading recent predictions:', error);
            this.showPredictionsError();
        }
    }
    
    updateRecentPredictions(predictions) {
        const container = document.getElementById('recentPredictions');
        if (!container) return;
        
        if (predictions.length === 0) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <i class="fas fa-inbox fa-3x text-muted mb-3"></i>
                    <h5>No Recent Predictions</h5>
                    <p class="text-muted">Start making predictions to see them here.</p>
                    <a href="/upload" class="btn btn-primary-custom">
                        <i class="fas fa-plus"></i> Make First Prediction
                    </a>
                </div>
            `;
            return;
        }
        
        const predictionsHtml = predictions.map(pred => `
            <div class="prediction-item fade-in">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <div class="prediction-result ${pred.result.toLowerCase()}">
                            <i class="fas fa-${pred.result.toLowerCase() === 'lvh' ? 'exclamation-circle' : 'check-circle'}"></i>
                            ${pred.result === 'LVH' ? 'LVH Detected' : 'Normal Result'}
                        </div>
                        <div class="prediction-meta">
                            Patient: ***${pred.patient_id_hash} | 
                            <span class="prediction-confidence">${(pred.confidence * 100).toFixed(1)}%</span> confidence |
                            <span class="prediction-data-type">${pred.data_types}</span>
                        </div>
                    </div>
                    <div class="text-end">
                        <div class="prediction-time">${pred.time_ago}</div>
                        <small class="text-muted">${pred.processing_time.toFixed(2)}s</small>
                    </div>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = predictionsHtml;
    }
    
    async loadModelPerformance() {
        try {
            const response = await fetch('/api/dashboard/model-performance');
            const data = await response.json();
            
            if (data.success) {
                this.updateModelPerformance(data.performance);
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error loading model performance:', error);
        }
    }
    
    updateModelPerformance(performance) {
        // Update ECG Model
        this.updateElement('ecgAccuracy', `${performance['ECG Model'].toFixed(1)}%`);
        this.updatePerformanceBar('ecgBar', performance['ECG Model']);
        
        // Update MRI Model
        this.updateElement('mriAccuracy', `${performance['MRI Model'].toFixed(1)}%`);
        this.updatePerformanceBar('mriBar', performance['MRI Model']);
        
        // Update Clinical Model
        this.updateElement('clinicalAccuracy', `${performance['Clinical Model'].toFixed(1)}%`);
        this.updatePerformanceBar('clinicalBar', performance['Clinical Model']);
        
        // Update Ensemble Model
        this.updateElement('ensembleAccuracy', `${performance['Ensemble'].toFixed(1)}%`);
        this.updatePerformanceBar('ensembleBar', performance['Ensemble']);
    }
    
    updatePerformanceBar(elementId, percentage) {
        const element = document.getElementById(elementId);
        if (element) {
            setTimeout(() => {
                element.style.width = `${percentage}%`;
            }, 500);
        }
    }
    
    initializeCharts() {
        const ctx = document.getElementById('accuracyChart');
        if (ctx) {
            this.accuracyChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    datasets: [{
                        label: 'Accuracy %',
                        data: [85, 88, 92, 89, 94, 91, 95],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#667eea',
                        pointBorderColor: '#ffffff',
                        pointBorderWidth: 2,
                        pointRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            min: 80,
                            max: 100,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.1)'
                            },
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(0, 0, 0, 0.1)'
                            }
                        }
                    },
                    elements: {
                        point: {
                            hoverRadius: 8
                        }
                    }
                }
            });
        }
    }
    
    updateAccuracyChart(trendData) {
        if (this.accuracyChart && trendData.length === 7) {
            const percentageData = trendData.map(val => (val * 100).toFixed(1));
            this.accuracyChart.data.datasets[0].data = percentageData;
            this.accuracyChart.update('active');
        }
    }
    
    calculateUptime() {
        const uptime = Date.now() - this.startTime;
        const hours = Math.floor(uptime / (1000 * 60 * 60));
        const minutes = Math.floor((uptime % (1000 * 60 * 60)) / (1000 * 60));
        
        if (hours > 24) {
            const days = Math.floor(hours / 24);
            return `${days}d ${hours % 24}h`;
        } else if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else {
            return `${minutes}m`;
        }
    }
    
    startAutoRefresh() {
        // Refresh metrics every 30 seconds
        this.metricsTimer = setInterval(() => {
            this.loadSystemMetrics();
        }, this.updateInterval);
        
        // Refresh predictions every 2 minutes
        this.predictionsTimer = setInterval(() => {
            this.loadRecentPredictions();
            this.loadAnalytics();
        }, this.updateInterval * 4);
    }
    
    stopAutoRefresh() {
        if (this.metricsTimer) {
            clearInterval(this.metricsTimer);
            this.metricsTimer = null;
        }
        
        if (this.predictionsTimer) {
            clearInterval(this.predictionsTimer);
            this.predictionsTimer = null;
        }
    }
    
    updateElement(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
        }
    }
    
    showError(message) {
        console.error(message);
        // Could show a toast notification here
    }
    
    showMetricsError() {
        const statusText = document.getElementById('statusText');
        if (statusText) {
            statusText.textContent = 'ERROR';
        }
    }
    
    showAnalyticsError() {
        // Show error state for analytics
        console.error('Analytics loading failed');
    }
    
    showPredictionsError() {
        const container = document.getElementById('recentPredictions');
        if (container) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <i class="fas fa-exclamation-triangle fa-2x text-warning mb-3"></i>
                    <h6>Failed to Load Predictions</h6>
                    <button class="btn btn-outline-primary btn-sm" onclick="dashboard.loadRecentPredictions()">
                        <i class="fas fa-retry"></i> Retry
                    </button>
                </div>
            `;
        }
    }
    
    destroy() {
        this.stopAutoRefresh();
        if (this.accuracyChart) {
            this.accuracyChart.destroy();
        }
    }
}

// Dark mode functionality
function initializeDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const darkModeIcon = document.getElementById('darkModeIcon');
    
    if (!darkModeToggle || !darkModeIcon) return;
    
    // Check for saved theme preference or default to light mode
    const currentTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    
    // Update icon based on current theme
    if (currentTheme === 'dark') {
        darkModeIcon.classList.remove('fa-moon');
        darkModeIcon.classList.add('fa-sun');
    }
    
    darkModeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        
        // Update icon
        if (newTheme === 'dark') {
            darkModeIcon.classList.remove('fa-moon');
            darkModeIcon.classList.add('fa-sun');
        } else {
            darkModeIcon.classList.remove('fa-sun');
            darkModeIcon.classList.add('fa-moon');
        }
        
        // Update chart colors if in dark mode
        if (dashboard && dashboard.accuracyChart) {
            const isDark = newTheme === 'dark';
            dashboard.accuracyChart.options.scales.y.grid.color = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
            dashboard.accuracyChart.options.scales.x.grid.color = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
            dashboard.accuracyChart.update();
        }
    });
}

// Global functions
function refreshPredictions() {
    if (window.dashboard) {
        window.dashboard.loadRecentPredictions();
    }
}

function initializeDashboard() {
    window.dashboard = new DashboardManager();
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.destroy();
    }
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DashboardManager, initializeDashboard, initializeDarkMode };
}