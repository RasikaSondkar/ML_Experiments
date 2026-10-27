// Client-side interactions for Algorithm Dashboard

document.addEventListener('DOMContentLoaded', function () {
    // 1-click Sample Input Loader
    const fillSampleBtn = document.getElementById('btn-fill-sample');
    if (fillSampleBtn) {
        fillSampleBtn.addEventListener('click', function () {
            try {
                const sampleDataRaw = this.getAttribute('data-sample');
                if (!sampleDataRaw) return;
                const sampleData = JSON.parse(sampleDataRaw);

                for (const [key, val] of Object.entries(sampleData)) {
                    const inputField = document.getElementById('field-' + key) ||
                                       document.querySelector(`[name="${key}"]`);
                    if (inputField) {
                        inputField.value = val;
                        inputField.dispatchEvent(new Event('change'));
                    }
                }

                // Show subtle toast or highlight
                fillSampleBtn.textContent = 'Sample Loaded';
                setTimeout(() => {
                    fillSampleBtn.textContent = 'Load Sample Values';
                }, 1200);
            } catch (err) {
                console.error("Error loading sample data:", err);
            }
        });
    }

    // Auto-scroll to prediction results if present
    const resultBox = document.getElementById('prediction-result-container');
    if (resultBox) {
        resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
