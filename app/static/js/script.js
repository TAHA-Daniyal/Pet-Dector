// Preview uploaded image before submitting
document.getElementById('imageUpload').addEventListener('change', function(event) {
  const preview = document.getElementById('preview');
  preview.innerHTML = ''; // Clear previous content

  const file = event.target.files[0];
  if (file) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.onload = () => URL.revokeObjectURL(img.src); // Clean up memory
    preview.appendChild(img);
  } else {
    preview.innerHTML = '<p>No image uploaded yet.</p>';
  }
});
