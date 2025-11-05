class SmartSpacesApp {
    constructor() {
        this.currentPlans = [];
        this.initEventListeners();
    }

    initEventListeners() {
        // Input type change
        document.querySelectorAll('input[name="inputType"]').forEach(radio => {
            radio.addEventListener('change', (e) => this.toggleInputType(e.target.value));
        });

        // File upload
        const fileInput = document.getElementById('photo');
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Generate button
        document.getElementById('generateBtn').addEventListener('click', () => this.generatePlan());
    }

    toggleInputType(type) {
        const dimensionsSection = document.getElementById('dimensionsSection');
        const uploadSection = document.getElementById('uploadSection');
        
        if (type === 'dimensions') {
            dimensionsSection.style.display = 'block';
            uploadSection.style.display = 'none';
        } else {
            dimensionsSection.style.display = 'none';
            uploadSection.style.display = 'block';
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        const uploadArea = document.querySelector('.upload-area');
        
        if (file) {
            if (!file.type.startsWith('image/')) {
                this.showError('Please select an image file');
                return;
            }
            
            if (file.size > 5 * 1024 * 1024) {
                this.showError('File size must be less than 5MB');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = (e) => {
                uploadArea.innerHTML = `
                    <div style="text-align: center;">
                        <i>📁</i>
                        <p>File selected: ${file.name}</p>
                        <p style="font-size: 0.8rem; color: #718096;">Click to change</p>
                    </div>
                `;
            };
            reader.readAsDataURL(file);
        }
    }

    async generatePlan() {
        // Get form values
        const bedrooms = parseInt(document.getElementById('bedrooms').value);
        const bathrooms = parseInt(document.getElementById('bathrooms').value);
        const inputType = document.querySelector('input[name="inputType"]:checked').value;
        
        // Validation
        if (!bedrooms || bedrooms < 1) {
            this.showError('Please enter a valid number of bedrooms (at least 1)');
            return;
        }
        
        if (bathrooms < 0) {
            this.showError('Please enter a valid number of bathrooms');
            return;
        }

        const requirements = { bedrooms, bathrooms };
        const data = { requirements, input_type: inputType, user_id: 'user_' + Date.now() };

        // Add dimension or image data
        if (inputType === 'dimensions') {
            const length = parseInt(document.getElementById('length').value);
            const width = parseInt(document.getElementById('width').value);
            
            if (!length || !width || length < 5 || width < 5) {
                this.showError('Please enter valid dimensions (minimum 5x5)');
                return;
            }
            
            data.length = length;
            data.width = width;
        } else {
            const fileInput = document.getElementById('photo');
            if (!fileInput.files[0]) {
                this.showError('Please upload a floor plan image');
                return;
            }
            
            try {
                data.image = await this.fileToBase64(fileInput.files[0]);
            } catch (error) {
                this.showError('Error processing image: ' + error.message);
                return;
            }
        }

        this.showLoading();
        
        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Server error');
            }

            if (result.success) {
                this.currentPlans = result.plans; // Store plans for details
                this.displayResults(result.plans, result.message);
            } else {
                throw new Error(result.error || 'Generation failed');
            }

        } catch (error) {
            this.showError('Failed to generate plans: ' + error.message);
            console.error('Generation error:', error);
        }
    }

    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
            reader.readAsDataURL(file);
        });
    }

    displayResults(plans, message) {
        const resultsSection = document.getElementById('resultsSection');
        
        if (!plans || plans.length === 0) {
            resultsSection.innerHTML = `
                <div class="error-message">
                    No plans were generated. Please try different inputs.
                </div>
            `;
            return;
        }

        let plansHTML = `
            <div class="success-message">
                ${message}
            </div>
            <div class="plan-options">
        `;

        plans.forEach((plan, index) => {
            const optionNum = index + 1;
            const roomStats = plan.room_stats || {};
            
            plansHTML += `
                <div class="plan-option">
                    <div class="plan-image">
                        ${plan.image ? 
                            `<img src="${plan.image}" alt="Floor Plan Option ${optionNum}">` : 
                            '<p>Image not available</p>'
                        }
                    </div>
                    <div class="plan-info">
                        <h4>Option ${optionNum}</h4>
                        <div class="plan-stats">
                            <span>Utilization: ${roomStats.utilization_rate || 'N/A'}%</span>
                            <span>Bedrooms: ${roomStats.bedroom || 0}</span>
                            <span>Bathrooms: ${roomStats.bathroom || 0}</span>
                        </div>
                    </div>
                </div>
            `;
        });

        plansHTML += '</div>';
        resultsSection.innerHTML = plansHTML;
    }

    showLoading() {
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.innerHTML = `
            <div class="loading">
                <div class="loading-spinner"></div>
                <p>AI is generating your floor plans...</p>
                <p style="font-size: 0.9rem; color: #718096;">This may take a few seconds</p>
            </div>
        `;
    }

    showError(message) {
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.innerHTML = `
            <div class="error-message">
                ${message}
            </div>
        `;
    }
}

// Initialize the app when the page loads
const app = new SmartSpacesApp();