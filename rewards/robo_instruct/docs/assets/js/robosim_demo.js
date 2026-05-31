let curr_idx = 0; // Start from the first slide (index 0)
const slides = document.querySelectorAll('.media-slide');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
function updateButtons() {
    prevBtn.disabled = curr_idx === 0;
    nextBtn.disabled = curr_idx === slides.length - 1;
    // Update classes for visual feedback
    prevBtn.classList.toggle('disabled', curr_idx === 0);
    nextBtn.classList.toggle('disabled', curr_idx === slides.length - 1);
}

function displayPrevious() {
    if (curr_idx > 0) {
        slides[curr_idx].style.display = 'none'; // Hide current slide
        curr_idx--; // Decrement index
        slides[curr_idx].style.display = 'block'; // Show new slide
        updateButtons();
    }
}

function displayNext() {
    if (curr_idx < slides.length - 1) {
        slides[curr_idx].style.display = 'none'; // Hide current slide
        curr_idx++; // Increment index
        slides[curr_idx].style.display = 'block'; // Show new slide
        updateButtons();
    }
}

function toggleContent(type) {
    var demoContent = document.getElementById('demoContent');
    var videoContent = document.getElementById('videoContent');

    if (type === 'demo') {
        demoContent.classList.add('show');
        videoContent.classList.remove('show');
    } else if (type === 'video') {
        videoContent.classList.add('show');
        demoContent.classList.remove('show');
    }
}

updateButtons(); // Initial button state check

