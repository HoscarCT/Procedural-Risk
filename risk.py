import cv2
import random
import numpy as np
from queue import Queue

height = 480
width = 720

class circle:
    def __init__ (self, x, y, r=3):
        self.x = x
        self.y = y
        self.r = r
    
    def __str__(self):
        return "({0},{1},{2})".format(self.x,self.y,self.r)

def point_with_distance(img, mask, N=50, min_d = 10, max_d = 50, tile_size = 10):
    centers = []
    
    i = 0
    while(i<1):
        radius = random.randint(min_d,max_d)
        x=random.randint(0,width/tile_size-1)
        y=random.randint(0,height/tile_size-1)
        x=int(tile_size/2)+tile_size*x
        y=int(tile_size/2)+tile_size*y
        if mask[y,x]:
            cv2.circle(img, (x,y), radius, (random.randint(100, 255),random.randint(100, 255),random.randint(100, 255)) , -1)
            centers.append(circle(x,y,radius))
            i+=1
        
    while(i<N):
        canPlace = True
        radius = random.randint(min_d,max_d)
        x=random.randint(0,width/tile_size-1)
        y=random.randint(0,height/tile_size-1)
        x=int(tile_size/2)+tile_size*x
        y=int(tile_size/2)+tile_size*y
        #print(x,y,i)
        if mask[y,x]:
            for center in centers:
                dist = center.r + radius
                if abs(x-center.x)<dist or abs(y-center.y)<dist:
                    dist_sq = (x-center.x)**2 + (y-center.y)**2
                    if (dist_sq < dist**2):
                        canPlace = False
                        break
            if (canPlace):
                centers.append(circle(x,y,radius))
                i+=1
                cv2.circle(img, (x,y), radius, (random.randint(100, 255),random.randint(100, 255),random.randint(100, 255)) , -1)
    return centers

def inBounds(i,j,image):
    return i>=0 and i<image.shape[1] and j>=0 and j<image.shape[0]

def tile_from_coord(y, x, tile_size):
    return int(y/tile_size)-int(tile_size/2),int(x/tile_size)-int(tile_size/2)

def grow_countries(tile_map, points, tile_size):

    points_list = [ [] for i in range(len(points)) ]
    points_queues = [ Queue() for i in range(len(points))]
    
    for i, p in enumerate(points):
        points_queues[i].put(p)
        cv2.circle(tile_map, (p.x,p.y), 2, (0,0,255), -1)

    allAreEmpty = False
    while not allAreEmpty:
        allAreEmpty = True
        for ii in range(len(points_queues)):
            if points_queues[ii].empty():
                continue
            else:
                allAreEmpty = False
            
            t = points_queues[ii].get()
            if random.random() < 0.8:
                points_list[ii].append(t)
                cv2.circle(tile_map, (t.x,t.y), 2, (0,0,255), -1)
                for i in range(t.x-tile_size, t.x+2*tile_size, tile_size):
                    for j in range(t.y-tile_size, t.y+2*tile_size, tile_size):
                        if inBounds(i,j,tile_map) and (i==t.x or j==t.y) and tile_map[j,i][1] == 255:
                            #print(tile_map[j,i])
                            if not (i==t.x and j==t.y):
                                    cv2.circle(tile_map, (i,j), 2, (255,0,0), -1)
                                    points_queues[ii].put(circle(i,j))
            else:
                points_queues[ii].put(t)
        
        cv2.imshow("REGION", tile_map)
        cv2.waitKey(20)
    
    return points_list

def neighbourLand(map_base, x, y):
    land = -map_base[y,x]
    for i in range(x-1, x+2):
        #print(i)
        for j in range(y-1, y+2):
            if inBounds(i,j,map_base):
                land+=map_base[j,i]
    return land

def smoothMap(map_base, terrain_factor):

    map2 = map_base.copy()
    height, width, _ = map_base.shape
    for i in range(height):
        for j in range(width):
            nLand = neighbourLand(map_base, j, i)
            if nLand > terrain_factor:
                #print("YES")
                map2[i,j] = 1
            elif nLand < terrain_factor:
                #print("NO---")
                map2[i,j] = 0

    return map2

def createMap(width, height, prob = 0.5, smoothing_iter = 5, terrain_factor = 4):

    map_base = np.zeros((height, width, 1), np.uint8)
    for i in range(height):
        for j in range(width):
            map_base[i,j] = 1 if random.random()<prob else 0

    for i in range(smoothing_iter):
        map_base = smoothMap(map_base, terrain_factor)
    
    map_base = map_base * 255
    return map_base

if __name__ == "__main__":
    
    #WORLD
    map_base = createMap(50, 25, 0.5, 3, 4)
    map_big = np.ones((height,width,1), np.uint8)
    cv2.resize(map_base, (width,height), map_big)
    _, mask = cv2.threshold(map_big, 50, 255, cv2.THRESH_BINARY)

    #TILES
    tile_size = 10
    tiles = np.zeros((height, width, 1), np.uint8)
    for i in range(0,int(height/tile_size)):
        for j in range(0,int(width/tile_size)):
            if(mask[int(tile_size/2)+int(i*tile_size),int(tile_size/2)+int(j*tile_size)]):
                cv2.circle(tiles, (int(tile_size/2)+int(j*tile_size),int(tile_size/2)+int(i*tile_size)), 2, 255, -1)
    
    #Calculate points
    img_circ = np.zeros((height,width,3), np.uint8)
    points = point_with_distance(img_circ, mask, 50, 20, 40, tile_size) #CUSTOM FUNCTION TO CREATE POINTS
    for p in points :
        cv2.circle(tiles, (p.x,p.y), 2, (100), -1)

    backtorgb = cv2.cvtColor(tiles, cv2.COLOR_GRAY2RGB)
    countries = grow_countries(backtorgb, points, tile_size)

    mapa = np.zeros((height,width,3), np.uint8)
    mapa[:, :] = (150,100,0)
    for country in countries:
        color = (random.randint(100,255), random.randint(100,255), random.randint(100,255))
        for p in country:
            cv2.circle(mapa, (p.x,p.y), 5, color, -1)

    cv2.imshow("THE WORLD", mapa)
    cv2.waitKey(0)

    # Show results
    #cv2.imshow("TILES", tiles)
    #cv2.imshow("CIRCLES",img_circ)
    #cv2.imshow("VORONOI",front)
    cv2.waitKey(0)

    

