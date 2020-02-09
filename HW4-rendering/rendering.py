# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:37:51 2020

@author: Antoine
"""

from PIL import Image
import numpy as np
from matplotlib import cm

def read_display_image(file_path, display = False):
    im = np.array(Image.open(file_path)).astype(float)
    im_normal = im[:, :, :3]
    
    # Normalize between 0 and 1 for PIL compatibility
    m = np.min(im_normal)
    im_normal = (im_normal - m)
    M = np.max(im_normal)
    im_normal = im_normal/M
    
    # Show and save for first question
    if display:
        im_display = Image.fromarray((im_normal*255).astype(np.uint8))
        im_display.show()
        im_display.save('render.png')
        
    return im_normal 

class LambertMaterial:
    def __init__(self, color, kd):
        self.color = np.array(color).reshape((1, 1, -1)) # Albedo color: base color of material
        self.kd = kd # Diffus Coefficient
    
    def fd(self):
        # Computes Lambert BRDF
        fd = self.kd * self.color / np.pi
        return(fd)

class BPMaterial:
    def __init__(self, ks, s):
        self.ks = ks # Specularity coefficient
        self.s = s # Shininess

    def fs(self, normal, wi, wo):
        # Computes Blinn-Phong BRDF
        wh = wi + wo
        wh = wh/np.linalg.norm(wh, axis = -1, keepdims = True)
        fs = self.ks * np.clip((normal*wh).sum(-1)[:, :, np.newaxis],0,1)**self.s
        return fs

class CTMaterial:
    def __init__(self, alpha, beta, color, n):
        self.alpha = alpha # Roughness
        self.beta = beta # Metallic property, False if dielectric, True if conductor
        self.color = np.array(color).reshape((1, 1, -1)) # Specular color
        self.n = n # Fresnel index

    def D(self, normal, wh):
        # Computes GGX as normal distribution function as in the course
        nwh = np.clip((normal*wh).sum(-1)[:, :, np.newaxis],0,1)
        d = np.pi * (1 + (self.alpha**2 - 1) * nwh**2)**2
        return (self.alpha)**2 / d

    def F(self, wi, wh):
        # Computes the spherical gaussian variant of the Schlick Fresnel approximation as in the course
        f0 = ((self.n-1)/(self.n+1))**2
        f0 = np.array([f0, f0, f0]).reshape((1, 1, -1)) if self.beta == False else self.color
        wiwh = np.clip((wi*wh).sum(-1)[:, :, np.newaxis],0,1)
        return f0 + (1 - f0) * (1 - wiwh)**5


    def G(self, normal, wi, wo):
        # Computes the Schlick approximation to the Smith model as in the course
        k = self.alpha * np.sqrt(2 / np.pi)
        nwi = np.clip((normal*wi).sum(-1)[:, :, np.newaxis],0,1)
        nwo = np.clip((normal*wo).sum(-1)[:, :, np.newaxis],0,1)
        gwi = nwi/(nwi*(1-k)+k)
        gwo = nwo/(nwo*(1-k)+k)
        return gwi * gwo


    def fs(self, normal, wi, wo):
        # Computes Cook-Terrance BRDF
        wh = wi + wo
        wh = wh/np.linalg.norm(wh, axis = -1, keepdims = True)
        D = self.D(normal, wh)
        F = self.F(wi, wh)
        G = self.G(normal, wi, wo)
        nwi = np.clip((normal*wi).sum(-1)[:, :, np.newaxis],0,1)
        nwo = np.clip((normal*wo).sum(-1)[:, :, np.newaxis],0,1)
        denom = 4 * nwi * nwo
        denom[denom == 0] = 1.
        fs =  D * F * G / denom
        return fs
        
class LightSource:
    def __init__(self, coord, color, intensity):
        # Coordinates, color and intensity of the Light Source
        self.coord = np.array(coord).reshape((1, 1, -1)) 
        self.color = np.array(color).reshape((1, 1, -1)) 
        self.intensity = intensity
    
    def wi(self, img_coord):
        # Computes incoming light direction
        wi = img_coord - self.coord
        # Normalize
        wi = wi / np.linalg.norm(wi, axis = -1, keepdims = True)
        return wi
    
    def Li(self):
        # Computes received light
        return(self.color*self.intensity)

def shade(normalImage, method = "Lambert"):
    light = LightSource([0., 1., 1.], [0.5, 0.7, 0.5], 1) # Color arbitrarly chosen
    
    Lambert_brdf = LambertMaterial([0.4, 0.7, 0.4], 1) # Color arbitrarly chosen
    if method == 'BP':
        fs_brdf = BPMaterial(1, 1)
    elif method == 'CT':
        fs_brdf = CTMaterial(0.7, True, [0.98, 0.82, 0.76], 1) 
    
    # Expand 2D image to 3D point cloud
    h, w, c = normalImage.shape
    X, Y = np.mgrid[0:h, 0:w]
    img_coords = np.stack((X, Y, np.zeros(X.shape)), axis = -1)
    
    # Computations
    wi = light.wi(img_coords)
    # Computes outgoing direction depending on camera coordinates
    camera_coords = [1.,300.,1.] # Camera position arbitrarly chosen
    camera_coords = np.array(camera_coords).reshape((1, 1, -1))
    wo = camera_coords - img_coords
    wo = wo/np.linalg.norm(wo, axis = -1, keepdims = True) 
    Li = light.Li()
    fd = Lambert_brdf.fd()
    fs = np.array([0]) if method == 'Lambert' else fs_brdf.fs(normalImage, wi, wo)
    
    # Computes Lo and normalize 
    nw = np.clip((wi*normalImage).sum(-1)[:, :, np.newaxis],0,1)
    Lo = Li * (fs + fd) * nw
    m, M = np.min(Lo), np.max(Lo)
    Lo = (Lo - m) / (M - m)
    
    return Lo

if __name__ == "__main__":
    # normalImage = read_display_image("normal.png", True)
    normalImage = read_display_image("normal.png")
    
    render_img = shade(np.array(normalImage), "Lambert")
    display_img= Image.fromarray((render_img * 255).astype(np.uint8))
    display_img.show()
    
    render_img = shade(np.array(normalImage), "BP")
    display_img = Image.fromarray((render_img * 255).astype(np.uint8))
    display_img.show()
    
    render_img = shade(np.array(normalImage), "CT")
    display_img = Image.fromarray((render_img * 255).astype(np.uint8))
    display_img.show()
    
    
        
    
    