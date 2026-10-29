
    const kamera = document.getElementById("kamera");
    const tombol = document.getElementById("mulai");

    tombol.addEventListener("click", async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });

        kamera.srcObject = stream;
      } catch (error) {
        alert("Kamera gagal dibuka. Periksa izin kamera browser.");
        console.error(error);
      }
    });
