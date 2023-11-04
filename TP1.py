# Importation des bibliothèques nécessaires
import cv2
import numpy as np
import matplotlib.pyplot as plt
from LibIP import imnoise

import warnings
warnings.filterwarnings("ignore")

# Fonction pour afficher une image avec OpenCV
def show_image(I, text):
    # Création d'une fenêtre avec le nom spécifié
    cv2.namedWindow(text)
    # Déplacement de la fenêtre à la position (20, 20) sur l'écran
    cv2.moveWindow(text, 20, 20)
    # Affichage de l'image 'I' dans la fenêtre
    cv2.imshow(text, I)
    
# Lecture des images
I = cv2.imread("Images/RGBA.png", -1) #Lire l'image sans la modifié
I_bw = cv2.imread("Images/RGBA.png", 0)	#Lire l'image noir & blanc
I_bricks = cv2.imread("Images/Bricks.jpg",0)
show_image(I, "RGBA image")


# =============================================================================
# Ex III - 1
# =============================================================================
# Affichage des dimensions de l'image
print("\nImage shape: ",I.shape)

# =============================================================================
# Ex III - 2
# =============================================================================
# Affichage des valeurs des pixels à des positions spécifiques
print("\nPixel 128,128. Rouge",I[128,128])
print("\nPixel 384,128. Vert",I[384,128])
print("\nPixel 128,384. Bleu",I[128,384])
print("\nPixel 384,384. Blanc",I[384,384])
print("\nPixel 1,1. Transparent",I[1,1])

# Affichage des couches de couleurs de l'image
show_image(I[:,:,0], "Blue color")
show_image(I[:,:,1], "Green color")
show_image(I[:,:,2], "Red color")
show_image(I[:,:,3], "Opacity")


# =============================================================================
# Ex III - 3
# =============================================================================
# Manipulation de la transparence de l'image
b, v, r ,t =cv2.split(I)
t_opaque=t/255
v_opaque=v*t_opaque
b_opaque=b*t_opaque
r_opaque=r*t_opaque
I[:,:,0]=b_opaque
I[:,:,1]=v_opaque
I[:,:,2]=r_opaque
I[:,:,3]=t_opaque

show_image(I, "RGBA image (opaque)")


# =============================================================================
# Ex III - 4
# =============================================================================
# Calcul de la luminance de l'image
y=0.299*r + 0.5884*v + 0.114*b


# =============================================================================
# Ex III - 5
# =============================================================================
# Sauvegarde de l'image en niveaux de gris
cv2.imwrite("Images/Niv_gris.png", y)


# =============================================================================
# Ex III - 6
# ============================================================================= 
# Affichage de l'image en niveaux de gris
show_image(I_bw, "Gray image")


# =============================================================================
# Ex IV - 2, 3
# =============================================================================
# Sous-échantillonnage de l'image et sauvegarde
I_echant = I_bw[::3,::3]
show_image(I_echant, "Image echantillonee")
cv2.imwrite("Petite_image.png", I_echant)


# =============================================================================
# Ex IV - 4
# =============================================================================
I_bricks_echant = I_bricks[::3,::3]
show_image(I_bricks_echant, "Bricks echantillonee")
cv2.imwrite("Petite_Bricks.jpg", I_bricks_echant)


# =============================================================================
# Ex IV - 6
# =============================================================================
# Redimensionnement des images
dim=(I_bw.shape[1]//3,I_bw.shape[0]//3)
I_resize = cv2.resize(I_bw,dim, interpolation=cv2.INTER_AREA)
show_image(I_resize, "Resized RGBA")
cv2.imwrite("Images/EX3_Q6_resize_RGBA.jpg", I_resize)

dim=(I_bricks.shape[1]//3,I_bricks.shape[0]//3)
I_bricks_resize =cv2.resize(I_bricks,dim, interpolation=cv2.INTER_AREA)
show_image(I_bricks_resize, "Resized Bricks")
cv2.imwrite("EX3_Q6_resize_Bricks.jpg", I_bricks_resize)


# =============================================================================
# Ex IV - 7
# =============================================================================
# Affichage des images redimensionnées avec Matplotlib
plt.figure()
plt.subplot(121)
plt.imshow(I_resize, 'gray'), plt.title('Resized RGBA')

plt.subplot(122)
plt.imshow(I_bricks_resize, 'gray'), plt.title('Resized Bricks')

plt.show()


# =============================================================================
# Ex V - 1, 2
# =============================================================================
# Application de différents filtres sur l'image
kernel = np.ones((20,20))/400
img_filter2d = cv2.filter2D(src=I_bw, ddepth=-1, kernel=kernel)

plt.figure()
plt.subplot(121)
plt.imshow(img_filter2d, 'gray'), plt.title('Linear filter')

img_median = cv2.medianBlur(I_bw,15)
plt.subplot(122)
plt.imshow(img_median, 'gray'), plt.title('Median filter')

plt.show()

# =============================================================================
# Ex V - 3
# =============================================================================
# Application de bruits sur l'image et filtrage
noise = imnoise(I_bw, V=1600)
plt.figure()
plt.subplot(221)
plt.imshow(noise, 'gray'), plt.title('Image with noise')

img_gaus_blur = cv2.GaussianBlur(noise,(15,15), sigmaX=3, sigmaY=3)
plt.subplot(222)
plt.imshow(img_gaus_blur, 'gray'), plt.title('Gaussian Blur')

kernel_gaus = cv2.getGaussianKernel(15, 3)
img_gaus = cv2.filter2D(noise, ddepth=-1, kernel=kernel_gaus)
plt.subplot(223)
plt.imshow(img_gaus, 'gray'), plt.title('Gaussian kernel')

img_median = cv2.medianBlur(noise,15)
plt.subplot(224)
plt.imshow(img_median, 'gray'), plt.title('Median blur')

plt.show()

# Calcul de l'erreur quadratique moyenne (MSE) entre deux images
def MSE_score(I_bw, img_gaus, text):
    sum = 0
    for n in range(I_bw.shape[0]):
        for m in range(I_bw.shape[1]):
            sum += (I_bw[n,m] - img_gaus[n,m])
    print(f'MSE {text} : ', sum/(I_bw.shape[0]*I_bw.shape[1]))

MSE_score(I_bw, img_gaus, 'Gaussian kernel')
MSE_score(I_bw, img_median, 'Median blur')


# =============================================================================
# Ex V - 4
# =============================================================================
# Application de bruit "sel & poivre" et filtrage
noise_sp = imnoise(I_bw, 's&p', V=1600)
plt.figure()
plt.subplot(221)
plt.imshow(noise_sp, 'gray'), plt.title('Image with salt & pepper')

dst = cv2.GaussianBlur(noise_sp,(15,15), sigmaX=3, sigmaY=3)
plt.subplot(222)
plt.imshow(dst, 'gray'), plt.title('Gaussian Blur')

kernel_gaus = cv2.getGaussianKernel(15, 3)
img_gaus = cv2.filter2D(noise_sp, ddepth=-1, kernel=kernel_gaus)
plt.subplot(223)
plt.imshow(img_gaus, 'gray'), plt.title('Gaussian kernel')

img_median = cv2.medianBlur(noise,15)
plt.subplot(224)
plt.imshow(img_median, 'gray'), plt.title('Median blur')

plt.show()

MSE_score(I_bw, img_gaus, 'Gaussian kernel')
MSE_score(I_bw, img_median, 'Median blur')

# =============================================================================
# Ex V - 5, 6
# =============================================================================
I_boat = cv2.imread("Images/boat.512.tiff", 0)

# Mesure du temps d'exécution des filtres
tic = cv2.getTickCount()
kernel = np.ones((20,20))/400
img_2d = cv2.filter2D(src=I_boat, ddepth=-1, kernel=kernel)
toc = cv2.getTickCount()
print('Filter Lin exec time =',( toc-tic )/cv2.getTickFrequency()*1000, 'ms')

plt.figure()
plt.subplot(121)
plt.imshow(img_2d, 'gray'), plt.title('Boat linear filter')
tic = cv2.getTickCount()
img_median = cv2.medianBlur(I_boat,15)
toc = cv2.getTickCount()
print('Filter Median exec time =',( toc-tic )/cv2.getTickFrequency()*1000, 'ms')

plt.subplot(122)
plt.imshow(img_median, 'gray'), plt.title('Boat median filter')

plt.show()


# =============================================================================
# Ex BONUS
# =============================================================================


# =============================================================================
# Ex VI - 1
# =============================================================================
# Transformée de Fourier de l'image
def fourier(image, m_text='Module', a_text='Argument'):
    f_transform = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transform)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(20*np.log(np.abs(f_shifted)), 'gray')
    plt.title(m_text)
    
    plt.subplot(122)
    plt.imshow(np.angle(f_shifted), 'gray')
    plt.title(a_text)

fourier(I_bricks, 'Module', 'Argument')


# =============================================================================
# Ex VI - 2
# =============================================================================
# Découpage de l'image
I_bricks_decoup = I_bricks[100:356, 150:450]
plt.figure()
plt.imshow(I_bricks_decoup, 'gray')
plt.title('Bricks découpé')

fourier(I_bricks_decoup, 'Module découpé', 'Argument découpé')


# =============================================================================
# Ex VI - 4
# =============================================================================
# Rotation de l'image
h, w = I_bricks_decoup.shape[:2]
center = (w/2, h/2)
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=60, scale=1)
rotated_image = cv2.warpAffine(I_bricks_decoup, rotate_matrix, (w, h))
show_image(rotated_image, 'Rotated image')


# =============================================================================
# Ex VI - 5
# =============================================================================
fourier(rotated_image, 'Module pivoté', 'Argument pivoté')


# =============================================================================
# Ex VI - 6
# =============================================================================
# Translation de l'image
tx, ty = 50, 50
translation_matrix = np.float32([[1, 0, tx],
                                [0, 1, ty]])
translated_image = cv2.warpAffine(I_bricks_decoup, translation_matrix, (w, h))
show_image(translated_image, 'Image decalee')
fourier(translated_image, 'Module décalé', 'Argument décalé')


# =============================================================================
# Ex VI - 7
# =============================================================================
# Transformée de Fourier du noyau gaussien
kernel_gaus = cv2.getGaussianKernel(15, 3)
fourier(kernel_gaus, 'Noyau gaussien', 'Argument gaussien')


# =============================================================================
# BONUS
# =============================================================================


# =============================================================================
# Wait for escape keypress
# =============================================================================
while(1):
    # Attente d'une touche pendant 1 seconde
    k = cv2.waitKey(1000)& 0xFF
    # Si la touche échappement (Esc) est appuyée
    if k ==27: #Esc key to stop
        break
# Fermeture de toutes les fenêtres créées par OpenCV
cv2.destroyAllWindows()
