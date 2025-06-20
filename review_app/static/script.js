document.addEventListener('DOMContentLoaded', () => {
    const galleryView = document.getElementById('gallery-view');
    const modalView = document.getElementById('modal-view');
    const finalizeButton = document.getElementById('finalize-button');
    let imagesData = [];

    // Base path for images relative to the HTML file
    // The images are now in `example-files/selected-pics/...` relative to the project root,
    // so we need to go up one level from the `static` folder.
    const imageBasePath = '../../example-files/selected-pics/';

    async function fetchData() {
        try {
            const response = await fetch('/api/images');
            imagesData = await response.json();
            renderGallery();
        } catch (error) {
            console.error('Failed to fetch image data:', error);
            galleryView.innerHTML = '<p>Error loading images. Is the server running?</p>';
        }
    }

    function renderGallery() {
        galleryView.innerHTML = '';
        imagesData.forEach(image => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.dataset.filename = image.filename;

            const statusIcon = image.is_approved ? '✔' : '❌';
            const statusClass = image.is_approved ? 'approved' : 'rejected';
            
            const actionButtonText = image.is_approved ? 'Reject' : 'Approve';
            const actionButtonClass = image.is_approved ? 'reject' : 'approve';

            const imagePath = `${imageBasePath}${image.filename.split('.')[0]}/featured.${image.filename.split('.')[1]}`;

            card.innerHTML = `
                <div class="image-card-thumbnail" style="background-image: url('${imagePath}')">
                    <span class="status-icon ${statusClass}">${statusIcon}</span>
                </div>
                <div class="image-card-info">
                    <h3>${image.title || 'Untitled'}</h3>
                    <div class="image-card-details">
                        <span>Score: ${image.quality_score}</span>
                        <button class="action-button ${actionButtonClass}">${actionButtonText}</button>
                    </div>
                </div>
            `;
            
            card.querySelector('.image-card-thumbnail').addEventListener('click', () => openModal(image));
            card.querySelector('h3').addEventListener('click', () => openModal(image));
            card.querySelector('.action-button').addEventListener('click', (e) => {
                e.stopPropagation();
                handleStatusToggle(image.filename, !image.is_approved);
            });

            galleryView.appendChild(card);
        });
    }

    async function handleStatusToggle(filename, newStatus) {
        try {
            const response = await fetch(`/api/update_status/${filename}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_approved: newStatus }),
            });
            if (response.ok) {
                const imageToUpdate = imagesData.find(img => img.filename === filename);
                if (imageToUpdate) {
                    imageToUpdate.is_approved = newStatus;
                    updateCard(imageToUpdate);
                    if (modalView.style.display !== 'none' && document.getElementById('modal-title').textContent === imageToUpdate.title) {
                        updateModalActions(imageToUpdate);
                    }
                }
            }
        } catch (error) {
            console.error('Failed to update status:', error);
        }
    }

    function updateCard(image) {
        const card = document.querySelector(`.image-card[data-filename="${image.filename}"]`);
        if (card) {
            const statusIconEl = card.querySelector('.status-icon');
            const actionButtonEl = card.querySelector('.action-button');
            
            statusIconEl.textContent = image.is_approved ? '✔' : '❌';
            statusIconEl.className = `status-icon ${image.is_approved ? 'approved' : 'rejected'}`;
            
            actionButtonEl.textContent = image.is_approved ? 'Reject' : 'Approve';
            actionButtonEl.className = `action-button ${image.is_approved ? 'reject' : 'approve'}`;
        }
    }

    function openModal(image) {
        const imagePath = `${imageBasePath}${image.filename.split('.')[0]}/featured.${image.filename.split('.')[1]}`;
        document.getElementById('modal-image').src = imagePath;
        document.getElementById('modal-title').textContent = image.title || 'Untitled';
        document.getElementById('modal-score-status').innerHTML = `<strong>Score:</strong> ${image.quality_score} | <strong>Status:</strong> ${image.is_approved ? 'Approved' : 'Rejected'}`;
        document.getElementById('modal-date').innerHTML = `<strong>Date:</strong> ${image.date_taken}`;
        document.getElementById('modal-summary').textContent = image.summary || 'No summary available.';
        document.getElementById('modal-description').textContent = image.image_description || 'No description available.';
        document.getElementById('modal-assessment').textContent = image.critical_assessment || 'No assessment available.';

        updateModalActions(image);
        modalView.style.display = 'flex';
    }
    
    function updateModalActions(image) {
        const modalActions = document.getElementById('modal-actions');
        const actionButtonText = image.is_approved ? 'Reject' : 'Approve';
        const actionButtonClass = image.is_approved ? 'reject' : 'approve';
        modalActions.innerHTML = `<button class="action-button ${actionButtonClass}">${actionButtonText}</button>`;
        modalActions.querySelector('.action-button').addEventListener('click', (e) => {
            e.stopPropagation();
            handleStatusToggle(image.filename, !image.is_approved);
        });
    }

    modalView.querySelector('.modal-close-btn').addEventListener('click', () => {
        modalView.style.display = 'none';
    });
    modalView.addEventListener('click', (e) => {
        if (e.target === modalView) {
            modalView.style.display = 'none';
        }
    });

    finalizeButton.addEventListener('click', async () => {
        if (confirm('Are you sure you want to finalize selections? This will close the review application.')) {
            try {
                await fetch('/api/finalize', { method: 'POST' });
                document.body.innerHTML = '<h1>Selections finalized. You can close this tab now.</h1>';
            } catch (error) {
                console.error('Failed to finalize selections:', error);
                alert('Could not contact server to finalize. Please check the console.');
            }
        }
    });

    fetchData();
});