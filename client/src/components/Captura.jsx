import React, { useState } from 'react';
import '../styles.css/captura.css'

function Captura() {

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [imagePreview, setImagePreview] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      setUploadStatus('Please select a file first.');
      return;
    }

    const formData = new FormData();
    console.log(selectedFile)
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:5000/subir', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        setUploadStatus('File uploaded successfully!');
      } else {
        setUploadStatus(`File upload failed 1: ${response.statusText}`);
      }
    } catch (error) {
      setUploadStatus(`File upload failed 2: ${error.message}`);
    }
  };

  return (
    <div className='area-form'>
      <form onSubmit={handleSubmit}>
        <div className='upload-file'>
          <div className='text-file'>Agregar Imagen</div>
          <input type="file" onChange={handleFileChange} className='input-file' accept="image/*" />
          {imagePreview && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <img 
                src={imagePreview} 
                alt="Vista previa" 
                style={{ 
                  width: '90%', 
                  height: 'auto',
                  maxHeight: '400px' }} />
            </div>
          )}
        </div>
        <div className='view'>
          <div className='text-file'>Subir Imagen</div>
          <button type="submit">Upload</button>
          <div className='response'>{uploadStatus}</div>
        </div>
      </form>
    </div>
  );
}

export default Captura;