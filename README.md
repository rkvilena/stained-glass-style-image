# Penerapan Stained-glass Style pada Citra dengan Teknik Pengolahan Citra

IF4073 Interpretasi dan Pengolahan Citra <br>
13520134 | Raka Wirabuana Ninagan

## Abstrak

Jendela dengan stained-glass style dulu menjadi populer di Eropa pada abad pertengahan akhir untuk digunakan sebagai hiasan pada katedral, gereja, balai kota, dan perumahan elit. Stained-glass style dapat diterapkan pada citra digital biasa menggunakan beberapa teknik pengolahan citra yaitu k-means clustering, gaussian blur, edge detection, dan color burn. Hasil implementasi pada makalah ini berhasil menerapkan stained-glass style pada citra masukan dengan beberapa kekurangan seperti kemunculan artefak dan kurangnya efek cahaya pada citra.

## Cara menggunakan

1. Pada lokasi terluar `root`, jalankan perintah `python main.py -in-file ../img/<image.jpg> -n-clusters <K>`
2. Perhatikan untuk mengganti `<image.jpg>` menjadi nama file dari gambar yang akan dieksekusi sehingga gambar perlu ditempatkan pada folder img.
3. Diperlukan beberapa library seperti `opencv`, `skimage`, `numpy`, dll.
