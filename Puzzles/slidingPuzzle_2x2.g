world: {}


center(world): { Q:[0 0 .1] }
table(center): { Q:[0 0 -.075], shape:ssBox, size:[.5 .4 .05 .01], color:[.8] }
        
bound0(center): { Q:[0 -.12 -.025], shape:ssBox, size:[.2125 .05 .05 .01], color:[.8]}
bound1(center): { Q:[0 +.12 -.025], shape:ssBox, size:[.2125 .05 .05 .01], color:[.8]}
bound2(center): { Q:[-.12 0 -.025], shape:ssBox, size:[.05 .2125 .05 .01], color:[.8]}
bound3(center): { Q:[+.12 0 -.025], shape:ssBox, size:[.05 .2125 .05 .01], color:[.8]}

        
box0: { X:[.05 -.05 .11], shape:ssBox, size:[.095 .095 .095 .01], color:[1 0 0], mass: .1}
box1: { X:[-.05 -.05 .11], shape:ssBox, size:[.095 .095 .095 .01], color:[0 1 0], mass: .1}
box2: { X:[.05 .05 .11], shape:ssBox, size:[.095 .095 .095 .01], color:[0 0 1], mass: .1}

base: { X: [0, 0, .2], multibody }
fake(base): {}
jointA1(fake): { joint: transX, limits: [-.25,.25], mass: .01 }
jointA2(jointA1): { joint: transY, limits: [-.25,.25], mass: .01 }
jointA3(jointA2): { joint: transZ, limits: [-.2,.2], mass: .01 }
wedge(jointA3): { joint: hingeZ, limits: [-3.2,3.2], shape: ssCvx, core:[-.01 0 0 .01 0 0 0 -.02 .1 0 .02 .1], size:[.005], color: [0, 1, 1], mass: .1 inertia: [0.02 0.02 0.02]}


# add a frame for a stationary camera
camera {X:[0 0 1.5]}