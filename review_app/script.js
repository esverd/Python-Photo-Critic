document.addEventListener('DOMContentLoaded', () => {
    const galleryView = document.getElementById('gallery-view');
    const modalView = document.getElementById('modal-view');
    const finalizeButton = document.getElementById('finalize-button');
    const reviewSummaryEl = document.getElementById('review-summary');
    let imagesData = [];

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

    function updateSummary() {
        const total = imagesData.length;
        const approvedCount = imagesData.filter(img => img.is_approved).length;
        reviewSummaryEl.textContent = `Reviewing ${total} photos | ${approvedCount} approved for final selection.`;
    }

    function renderGallery() {
        galleryView.innerHTML = '';
        imagesData.forEach(image => {
            const card = document.createElement('div');
            card.className = 'bg-white rounded-xl shadow-lg overflow-hidden transition-all hover:shadow-2xl group/card';
            card.dataset.filename = image.filename;

            const score = parseFloat(image.quality_score).toFixed(1);
            const statusClass = image.is_approved ? 'bg-green-500' : 'bg-red-500';
            const statusText = image.is_approved ? 'Approved' : 'Rejected';

            // IMPORTANT: This now points to the new server endpoint for original images
            const imagePath = `/images/${image.filename}`;
            
            card.innerHTML = `
                <div class="relative">
                    <div class="w-full bg-center bg-no-repeat aspect-[3/2] bg-cover" style="background-image: url('${imagePath}');"></div>
                    <div class="absolute top-2 right-2 bg-black/60 text-white px-2 py-1 rounded-md text-xs font-semibold">AI Score: ${score}</div>
                    <div class="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/70 to-transparent opacity-0 group-hover/card:opacity-100 transition-opacity duration-300 flex justify-end items-center gap-2">
                        <button data-action="reject" class="p-2 rounded-full bg-red-500/80 hover:bg-red-500 text-white transition-colors" title="Reject">
                            <span class="material-symbols-outlined text-xl">thumb_down</span>
                        </button>
                        <button data-action="approve" class="p-2 rounded-full bg-green-500/80 hover:bg-green-500 text-white transition-colors" title="Approve">
                            <span class="material-symbols-outlined text-xl">thumb_up</span>
                        </button>
                    </div>
                </div>
                <div class="p-4">
                    <h3 class="text-gray-800 text-base font-semibold truncate" title="${image.title || 'Untitled'}">${image.title || 'Untitled'}</h3>
                    <div class="flex items-center justify-between mt-2">
                        <p class="text-gray-500 text-xs">Quality: ${score}</p>
                        <span class="status-badge inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-white ${statusClass}">${statusText}</span>
                    </div>
                </div>
            `;
            
            card.addEventListener('click', () => openModal(image));
            card.querySelector('[data-action="approve"]').addEventListener('click', (e) => {
                e.stopPropagation();
                handleStatusToggle(image.filename, true);
            });
            card.querySelector('[data-action="reject"]').addEventListener('click', (e) => {
                e.stopPropagation();
                handleStatusToggle(image.filename, false);
            });

            galleryView.appendChild(card);
        });
        updateSummary();
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
                    const card = galleryView.querySelector(`[data-filename="${imageToUpdate.filename}"]`);
                    if (card) {
                        const badge = card.querySelector('.status-badge');
                        badge.textContent = imageToUpdate.is_approved ? 'Approved' : 'Rejected';
                        badge.className = `status-badge inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-white ${imageToUpdate.is_approved ? 'bg-green-500' : 'bg-red-500'}`;
                    }
                    updateSummary();
                }
            }
        } catch (error) {
            console.error('Failed to update status:', error);
        }
    }

    function openModal(image) {
        modalView.style.display = 'flex';
        const imagePath = `/images/${image.filename}`;
        const score = parseFloat(image.quality_score).toFixed(1);

        modalView.innerHTML = `
            <div class="modal-content bg-white w-full max-w-6xl max-h-[90vh] rounded-lg relative flex flex-col">
                <button class="modal-close-btn absolute top-2 right-4 text-4xl text-gray-500 hover:text-gray-800">&times;</button>
                <div class="modal-body flex p-6 overflow-y-auto">
                    <div class="modal-image-container flex-shrink-0 w-3/5 pr-6">
                        <img id="modal-image" src="${imagePath}" class="w-full h-auto rounded-md object-contain max-h-[80vh]">
                    </div>
                    <div class="modal-details-container w-2/5">
                        <h4 class="text-2xl font-bold text-gray-800">${image.title || 'Untitled'}</h4>
                        <p class="mt-1"><strong>Score:</strong> ${score} | <strong>Date:</strong> ${image.date_taken}</p>
                        <div class="mt-4">
                            <h5 class="font-bold border-b pb-1 mb-2">Summary</h5>
                            <p class="text-gray-700">${image.summary || 'N/A'}</p>
                        </div>
                        <div class="mt-4">
                            <h5 class="font-bold border-b pb-1 mb-2">Objective Description</h5>
                            <blockquote class="text-sm text-gray-600 italic border-l-4 pl-4">${image.image_description || 'N/A'}</blockquote>
                        </div>
                        <div class="mt-4">
                            <h5 class="font-bold border-b pb-1 mb-2">Critical Assessment</h5>
                            <blockquote class="text-sm text-gray-600 italic border-l-4 pl-4">${image.critical_assessment || 'N/A'}</blockquote>
                        </div>
                    </div>
                </div>
            </div>
        `;

        modalView.querySelector('.modal-close-btn').addEventListener('click', () => {
            modalView.style.display = 'none';
        });
    }

    finalizeButton.addEventListener('click', async () => {
        if (confirm('Are you sure you want to finalize selections? This will close the review application.')) {
            try {
                await fetch('/api/finalize', { method: 'POST' });
                document.body.innerHTML = '<div class="w-screen h-screen flex items-center justify-center"><h1 class="text-2xl font-bold text-gray-800">Selections finalized. You can close this browser tab.</h1></div>';
            } catch (error) {
                console.error('Failed to finalize selections:', error);
                alert('Could not contact server to finalize. Please check the console.');
            }
        }
    });
    
    modalView.addEventListener('click', (e) => {
        if (e.target === modalView) {
            modalView.style.display = 'none';
        }
    });

    fetchData();
});