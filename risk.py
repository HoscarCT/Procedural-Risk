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

#--------------CAPITALS POSITIONING------------------

def random_circle_in_grid(min_d,max_d, width, height, tile_size):

    radius = random.randint(min_d,max_d)
    x=random.randint(0,width/tile_size-1)
    y=random.randint(0,height/tile_size-1)

    x=int(tile_size/2)+tile_size*x
    y=int(tile_size/2)+tile_size*y

    return circle(x,y,radius)

#This function could be optimized for the grid, but it was first developed for gridless maps
#Also, 'img' could be unused if visualization of the disks is not showed, in the final version, it isn't
def point_with_distance(img, mask, N=50, min_d = 10, max_d = 50, tile_size = 10):
    centers = []
    i = 0
    #Place first circle
    while(i<1):
        circ = random_circle_in_grid(min_d,max_d, width, height, tile_size)
        if mask[circ.y,circ.x]:
            cv2.circle(img, (circ.x,circ.y), circ.r, (random.randint(100, 255),random.randint(100, 255),random.randint(100, 255)) , -1)
            centers.append(circ)
            i+=1
    #Place the rest
    while(i<N):
        canPlace = True
        circ = random_circle_in_grid(min_d,max_d, width, height, tile_size)
        if mask[circ.y,circ.x]:
            for center in centers:
                dist = center.r + circ.r
                if abs(circ.x-center.x)<dist or abs(circ.y-center.y)<dist:      #If there's a possible collision
                    dist_sq = (circ.x-center.x)**2 + (circ.y-center.y)**2       #Calculate distance squared (no need to calculate the root)
                    if (dist_sq < dist**2):
                        canPlace = False
                        break                                                   #If 1 disk collides, break the loop
            if (canPlace):
                cv2.circle(img, (circ.x,circ.y), circ.r, (random.randint(100, 255),random.randint(100, 255),random.randint(100, 255)) , -1)
                centers.append(circ)
                i+=1
    return centers

#-------------COUNTRIES GENERATION----------------

def inBounds(i,j,image):
    return i>=0 and i<image.shape[1] and j>=0 and j<image.shape[0]

#Region grow to create countries from their capitals
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
            #Termination condition
            if points_queues[ii].empty():
                continue
            allAreEmpty = False
            
            t = points_queues[ii].get()
            if random.random() < 0.8:                                               #Randomize possiblity of exploring around
                points_list[ii].append(t)
                cv2.circle(tile_map, (t.x,t.y), 2, (0,0,255), -1)                   #Paint appended points blue
                for i in range(t.x-tile_size, t.x+2*tile_size, tile_size):
                    for j in range(t.y-tile_size, t.y+2*tile_size, tile_size):      #Look for adjacent points to conquer
                        if inBounds(i,j,tile_map) and (i==t.x or j==t.y) and tile_map[j,i][1] == 255:
                            if not (i==t.x and j==t.y):
                                    cv2.circle(tile_map, (i,j), 2, (255,0,0), -1)
                                    points_queues[ii].put(circle(i,j))              #Add the points to the region queue 
            else:
                points_queues[ii].put(t)
        
        cv2.imshow("REGION", tile_map)                                              #Show progress (I like watching the animation on loop '(^v^) )
        cv2.waitKey(20)
    
    return points_list

#----------WATER-LAND MASK GENERATION------------

#Returns number of neighbours
def neighbourLand(map_base, x, y):
    land = -map_base[y,x]
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if inBounds(i,j,map_base):
                land+=map_base[j,i]
    return land

#Function that returns a new smoothed map
def smoothMap(map_base, terrain_factor):
    map2 = map_base.copy()
    map_height, map_width, _ = map_base.shape
    for i in range(map_height):
        for j in range(map_width):
            nLand = neighbourLand(map_base, j, i)
            if nLand > terrain_factor:
                map2[i,j] = 1
            elif nLand < terrain_factor:
                map2[i,j] = 0
    return map2

def createMap(width, height, prob = 0.5, smoothing_iter = 5, terrain_factor = 4):

    map_base = np.zeros((height, width, 1), np.uint8)
    for i in range(height):
        for j in range(width):
            map_base[i,j] = 1 if random.random()<prob else 0                    #Initialize random pixels

    for i in range(smoothing_iter):
        map_base = smoothMap(map_base, terrain_factor)                          #Smooth based on neighbours

    map_base = map_base * 255                                                   #Scale colors to a visible image
    return map_base

if __name__ == "__main__":
    
    #MASK
    map_base = createMap(50, 25, 0.5, 3, 4)                                     #Try different dimensions for more squashed or stretched maps
    map_big = np.ones((height,width,1), np.uint8)
    cv2.resize(map_base, (width,height), map_big)
    _, mask = cv2.threshold(map_big, 50, 255, cv2.THRESH_BINARY)

    #TILES (This generates blank map with the tiled style based on the world mask)
    tile_size = 10
    tiles = np.zeros((height, width, 1), np.uint8)
    for i in range(0,int(height/tile_size)):
        for j in range(0,int(width/tile_size)):
            if(mask[int(tile_size/2)+int(i*tile_size),int(tile_size/2)+int(j*tile_size)]):
                cv2.circle(tiles, (int(tile_size/2)+int(j*tile_size),int(tile_size/2)+int(i*tile_size)), 2, 255, -1)
    
    #Calculate points
    img_circ = np.zeros((height,width,3), np.uint8)                             #Image to place the capitales
    points = point_with_distance(img_circ, mask, 50, 20, 40, tile_size)         #This function returns the list of capitals
    for p in points :
        cv2.circle(tiles, (p.x,p.y), 2, (100), -1)                              #For visualization

    #Grow countries from their capitals
    backtorgb = cv2.cvtColor(tiles, cv2.COLOR_GRAY2RGB)
    countries = grow_countries(backtorgb, points, tile_size)

    #Final map
    world = np.zeros((height,width,3), np.uint8)
    world[:, :] = (150,100,0)                                                    #Draw sea (change color to taste)
    for country in countries:
        color = (random.randint(100,255), random.randint(100,255), random.randint(100,255))     #This can be changed to draw color based on a random number
        for p in country:                                                                       #If you desire to paint the map with players already in place
            cv2.circle(world, (p.x,p.y), 5, color, -1)

    cv2.imshow("THE WORLD", world)
    cv2.imwrite("risk_map.png", world)
    cv2.waitKey(0)

    

