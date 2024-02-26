world: {}


center(world): { Q:[0 0 .1] }
table(center): { Q:[0 0 -.075], shape:ssBox, size:[.5 .5 .05 .01], color:[.8] }
        
bound0(center): { Q:[0 -.17 -.025], shape:ssBox, size:[.3 .05 .05 .01], color:[.8]}
bound1(center): { Q:[0 +.17 -.025], shape:ssBox, size:[.3 .05 .05 .01], color:[.8]}
bound2(center): { Q:[-.17 0 -.025], shape:ssBox, size:[.05 .3 .05 .01], color:[.8]}
bound3(center): { Q:[+.17 0 -.025], shape:ssBox, size:[.05 .3 .05 .01], color:[.8]}

        
box0: { X:[.1 -.1 .11], shape:ssBox, size:[.096 .096 .096 .01], color:[1 0 0], mass: .1}
box1: { X:[0. -.1 .11], shape:ssBox, size:[.096 .096 .096 .01], color:[0 1 0], mass: .1}
box2: { X:[-.1 -.1 .11], shape:ssBox, size:[.096 .096 .096 .01], color:[0 0 1], mass: .1}
box3: { X:[.1 0. .11], shape:ssBox, size:[.096 .096 .096 .01], color:[1 1 0], mass: .1}
box4: { X:[0. 0. .11], shape:ssBox, size:[.096 .096 .096 .01], color:[0 1 1], mass: .1}
box5: { X:[-.1 0. .11], shape:ssBox, size:[.096 .096 .096 .01], color:[1 0 1], mass: .1}
box6: { X:[.1 .1 .11], shape:ssBox, size:[.096 .096 .096 .01], color:[1 1 1], mass: .1}
box7: { X:[0. .1 .11], shape:ssBox, size:[.096 .096 .096 .01], color:[0 0 0], mass: .1}


#base { X:[0 0 .2], motors }
#fake(base){}
#jointA1(fake){ joint:transX, limits:[-.5,.5], mass:.01 }
#jointA2(jointA1){ joint:transY, limits:[-.5,.5], mass:.01 }
#jointA3(jointA2){ joint:transZ, limits:[-.5,.5], mass:.01 }
## made wide part of wedge smaller, because it was difficult to only push one cube
#wedge(jointA3): { joint:hingeZ, limits:[-3.2,3.2], shape:ssCvx, core:[-.01 0 0 .01 0 0 0 -.02 .1 0 .02 .1], size:[.005], color:[0 1 1], mass:.1 }

base: { X: [0, 0, .2], multibody}
fake(base): {}
jointA1(fake): { joint: transX, limits: [-.5,.5], mass: .01 }
jointA2(jointA1): { joint: transY, limits: [-.5,.5], mass: .01 }
jointA3(jointA2): { joint: transZ, limits: [-.5,.5], mass: .01 }
wedge(jointA3): { joint: hingeZ, limits: [-3.2,3.2], shape: ssCvx, core:[-.01 0 0 .01 0 0 0 -.02 .1 0 .02 .1], size:[.005], color: [0, 1, 1], mass: .1 inertia: [0.02 0.02 0.02]}

# add a frame for a stationary camera
camera {X:[0 0 1.5]}