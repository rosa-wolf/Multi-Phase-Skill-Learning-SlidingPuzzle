world: {}


center(world): { Q:[0 0 .1] }
table(center): { Q:[0 0 -.075], shape:ssBox, size:[.5 .4 .05 .01], color:[.8] }
        
bound0(center): { Q:[0 -.13 -.025], shape:ssBox, size:[.3125 .05 .05 .01], color:[.8] }
bound1(center): { Q:[0 +.13 -.025], shape:ssBox, size:[.3125 .05 .05 .01], color:[.8] }
bound2(center): { Q:[-.18 0 -.025], shape:ssBox, size:[.05 .3125 .05 .01], color:[.8] }
bound3(center): { Q:[+.18 0 -.025], shape:ssBox, size:[.05 .3125 .05 .01], color:[.8] }

        
box0: { X:[-.1 -.05 .11], shape:ssBox, size:[.1 .1 .1 .01], color:[1 1 0], mass: .1 }
box1: { X:[  0 -.05 .11], shape:ssBox, size:[.1 .1 .1 .01], color:[1 0 1], mass: .1 }
box2: { X:[ .1 -.05 .11], shape:ssBox, size:[.1 .1 .1 .01], color:[1 .5 0], mass: .1 }
box3: { X:[-.1 +.05 .11], shape:ssBox, size:[.1 .1 .1 .01], color:[.5 1 0], mass: .1 }
box4: { X:[  0 +.05 .11], shape:ssBox, size:[.1 .1 .1 .01], color:[0 .5 1], mass: .1 }

base { X:[0 0 .3], motors }
fake(base){}
jointA1(fake){ joint:transX, limits:[-.5,.5], mass:.01 }
jointA2(jointA1){ joint:transY, limits:[-.5,.5], mass:.01 }
jointA3(jointA2){ joint:transZ, limits:[-.5,.5], mass:.01 }
# made wide part of wedge smaller, because it was difficult to only push one cube
wedge(jointA3): { joint:hingeZ, limits:[-3.2,3.2], shape:ssCvx, core:[-.035 0 0 .035 0 0 0 -.02 .1 0 .02 .1], size:[.005], color:[0 1 1], mass:.1 }


# add a frame for a stationary camera
camera {X:[0 0 1.5]}