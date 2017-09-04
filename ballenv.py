import sys,pygame
from pygame.locals import *
from numpy.random import *
from numpy import *
import numpy as np
from math import * 

try:
    xrange = xrange
except:
    xrange = range

GODPOWER = 0.5
GODWALL = 50
OMOMI_KEISUU = 10

EYESHAPE=[65,65]
class distance:
    x = 100
    y = 100
    xint = 0
    yint = 0

class ball(pygame.sprite.Sprite):
    x = 0
    y = 0
    xint = 0
    yint = 0
    px = 0
    py = 0
    sx = 0.01
    sy = 0.01
    count = 0 
    size = 0
    color = (0,0,0)
    cal = 1
    power = 2
    protected = False
    #ball.screen
    
    
    def __init__(self,x=False,y=False,power=2,span=False,color=array([0,0,0])):
        pygame.sprite.Sprite.__init__(self,self.containers)
        if power:
            self.power =fabs( normal(power,10))
        self.size = randint(5,10)
        size_x = int(sin(pi/4)*self.size)
        size_y = int(cos(pi/4)*self.size)
        self.rectsize = (size_x,size_y)
        #self.color = (rand(255),rand(255),rand(255))
        if color.all():
            self.color = normal(color,10,3)
            for t in range(self.color.size):
                if self.color[t] < 0:
                    self.color[t] = 0
                elif self.color[t] > 255:
                    self.color[t] = 255
        else:
            self.color = randint(0,255,3)
        if x and y:
            self.x = x
            self.y = y
        else:
            self.x = randint(self.screen.width)
            self.y = randint(self.screen.height)
        #self rectangle size
        self.rect = Rect(self.x,self.y,self.rectsize[0],self.rectsize[1])
        #move dist update span
        if span:
            self.span = fabs(normal(span,20))
        else:
            self.span = randint(32,512)
        
        #pdb.set_trace()
    def updatedest(self):
        self.x = self.x + self.sx
        self.y = self.y + self.sy
        self.xint = int(self.x)
        self.yint = int(self.y)
        self.rect = Rect(self.xint,self.yint,self.rectsize[0],self.rectsize[1])
        """
        if(self.xint < 0 or self.xint > windowSize[0]):
            
            self.sx =0.1*  self.sx/abs(self.sx)
        if(self.yint < 0 or self.yint > windowSize[1]):
            #self.sy = -self.sy * 0.9
            self.py =0.1*  self.sy/abs(self.sy)
        """

    def updatespeed(self):
        self.count += 1
        if(self.count > self.span):
            self.px = (rand() - 0.5) / self.power
            self.py = (rand() - 0.5) / self.power
            self.count = 0

        if(self.rect.left<= 0+GODWALL):
            #self.xint = 0
            self.px =  GODPOWER
        if(self.rect.right >= self.screen.width -GODWALL):
            #self.xint = windowSize[0]
            self.px =  -GODPOWER

        if(self.rect.top <= 0+GODWALL):
            #self.yint = 0
            self.py =  GODPOWER
        if(self.rect.bottom >= self.screen.height -GODWALL):
            #self.yint = windowSize[1]
            self.py =  -GODPOWER

        self.sx = self.sx + self.px/(self.size*5)
        self.sy = self.sy + self.py/(self.size*5)

        #self.sx *= (1-fabs(self.sx))
        #self.sy *= (1-fabs(self.sy))
        self.sx *= 0.99
        self.sy *= 0.99
        #trying to kuukiteikou 
        #self.sx = 1.4 * tanh(self.sx)
        #self.sy = 1.4 * tanh(self.sy)

        """
        if(fabs(self.sx) > 1):
            self.px = 0
        if(fabs(self.sy) > 1):
            self.py = 0
        """
    def updatecollide(self):
        #collided = pygame.sprite.groupcollide(group ,group,False,False)
        collided = pygame.sprite.spritecollide(self ,self.containers,False)
        if collided:
            for collideball in collided:
                if((collideball.size/OMOMI_KEISUU+fabs(collideball.sx)+fabs(collideball.sy) < self.size/OMOMI_KEISUU+fabs(self.sx)+fabs(self.sy))) and collideball.protected == False:
                    self.containers.remove(collideball) 
                    self.size += collideball.size
                    self.cal += collideball.cal 
                    size_x = int(sin(pi/4)*self.size)
                    size_y = int(cos(pi/4)*self.size)
                    self.rectsize = (size_x,size_y)
    def update(self,screen):
        self.updatecollide()
        self.updatespeed()
        self.updatedest()
        #self.drawcircle(screen)
        self.drawrect(screen)
        #self.rect.move_ip(self.vx,self.vy)
        #self.lifespan += 1
        if self.size > 50:
            self.containers.remove(self)
            for i in range(self.cal):
                x = normal(self.rect.centerx,50)
                y = normal(self.rect.centery,50)
                ball(x,y,self.power,self.span,self.color) 

    def drawcircle(self,screen):
        pygame.draw.circle(screen,self.color,(self.xint,self.yint),self.size)

    def drawrect(self,screen):
        #pygame.draw.rect(screen,self.color,(self.xint,self.yint)+self.rectsize)
        screen.fill(self.color,self.rect)

#######################
# Eye Class def
#######################

class eye(pygame.sprite.Sprite):
    ''' 
    indexlist = [[(-3,-3),(-2,-3),(-1,-3),(0,-3),(1,-3),(2,-3),(3,-3)],
                [(-3,-2),(-2,-2),(-1,-2),(0,-2),(1,-2),(2,-2),(3,-2)],
                [(-3,-1),(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),(3,-1)],
                [(-3,0),(-2,0),(-1,0),(0,0),(1,0),(2,0),(3,0)],
                [(-3,1),(-2,1),(-1,1),(0,1),(1,1),(2,1),(3,1)],
                [(-3,2),(-2,2),(-1,2),(0,2),(1,2),(2,2),(3,2)],
                [(-3,3),(-2,3),(-1,3),(0,3),(1,3),(2,3),(3,3)]]
    '''
    shape = EYESHAPE
    reshaped = [int((shape[0] -1)/2), int((shape[1] -1)/2)]
    

    color = array([0,0,0])
    protected = True
    def __init__(self, x, y, size, index):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.indexlist = [[(i,j) for i in range(-self.reshaped[0],self.reshaped[0] + 1)] for j in range(-self.reshaped[1],self.reshaped[1]+1)] 
        self.size = size
        self.index = index
        size_x = int(sin(pi/4)*self.size)
        size_y = int(cos(pi/4)*self.size)
        self.rectsize = (size_x, size_y)
        self.shift = self.indexlist[index[0]][index[1]]
        self.x = x + size_x * self.shift[0]
        self.y = y + size_y * self.shift[1]
        self.rect = Rect(self.x, self.y, *self.rectsize)
    def update(self,screen,x,y):
        self.updatedest(x,y)
        self.drawrect(screen)
    def updatedest(self, x, y):
        self.x = x + self.rectsize[0] * self.shift[0]
        self.y = y + self.rectsize[1] * self.shift[1]
        self.rect = Rect(self.x, self.y, *self.rectsize)
    def drawrect(self,screen):
        screen.fill(self.color,self.rect)

####################################
# reinforcement learning ball Def    
####################################
class rlball(ball):

    power = 1
    size = 20
    x = 200
    y = 200
    span = 100
    protected = True
    eyesView = zeros(EYESHAPE)
    indexes = [[(x,y) for y in range(EYESHAPE[0])] for x in range(EYESHAPE[1])]
    eyesGroup = pygame.sprite.Group()
    selfeye = eye
    selfeye.containers = eyesGroup
    eyes = []
    # [x,y]
    action_list = [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]]

    def __init__(self,x=False,y=False,power=2,span=False,color=array([255,0,0])):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.x = randint(self.screen.width)
        self.y = randint(self.screen.height)
        size_x = int(sin(pi/4)*self.size)
        size_y = int(cos(pi/4)*self.size)
        self.rectsize = (size_x,size_y)
        self.rect = Rect(self.x,self.y,*self.rectsize)
        eye.screen = self.screen
        self.color = color
        #pdb.set_trace()
        for indexPre in self.indexes:
            for index in indexPre:
                if index == (int((EYESHAPE[0]-1)/2),int((EYESHAPE[1]-1)/2)):
                    continue
                self.eyes.append(eye(x,y,self.size,index))

    def isupdatable(self):
        updatable = False
        if(self.count > self.span):
            updatable = True
        return updatable                       
      
    def updatespeed(self):
        # overlide
        self.count += 1
        if(self.count > self.span):
            # FIXME define rand range
            #self.px = (rand() - 0.5) / self.power
            #self.py = (rand() - 0.5) / self.power
            # TODO action should be 1x8 size
            #action = agent.get_action(self.eyesView)
            #select_action = self.action_list[np.argmax(self.action)]
            #print(self.action)
            select_action = self.action_list[self.action]
            self.px = (select_action[0] ) / self.power
            self.py = (select_action[1] ) / self.power
            self.count = 0
            

        if(self.rect.left<= 0+GODWALL):
            #self.xint = 0
            self.px =  GODPOWER
        if(self.rect.right >= self.screen.width -GODWALL):
            #self.xint = windowSize[0]
            self.px =  -GODPOWER

        if(self.rect.top <= 0+GODWALL):
            #self.yint = 0
            self.py =  GODPOWER
        if(self.rect.bottom >= self.screen.height -GODWALL):
            #self.yint = windowSize[1]
            self.py =  -GODPOWER

        self.sx = self.sx + self.px/(self.size*5)
        self.sy = self.sy + self.py/(self.size*5)

        #self.sx *= (1-fabs(self.sx))
        #self.sy *= (1-fabs(self.sy))
        self.sx *= 0.99
        self.sy *= 0.99
        #trying to kuukiteikou 
        #self.sx = 1.4 * tanh(self.sx)
        #self.sy = 1.4 * tanh(self.sy)
  
    def updatecollide(self):
        self.collided = False
        self.reward = 0
        #pdb.set_trace()
        # collision detection self vs other
        collided = pygame.sprite.spritecollide(self,self.enemygroup,True)
        if collided:
            self.collided = True
            for collideball in collided:
                self.reward = self.reward + collideball.cal
        
        # collision detection Eye vs other
        collideEye = pygame.sprite.groupcollide(self.eyesGroup,self.enemygroup,False,False)
        self.eyesView = zeros(EYESHAPE)
        # built state(vision)
        for oneEye in collideEye:
            self.eyesView[oneEye.index] = 1
        #debug
        # print(self.eyesView)
           
    def update(self,screen):
        #pdb.set_trace()
        self.updatecollide()
        self.updatespeed()
        self.updatedest()
        # update eyes to following ball
        self.eyesGroup.update(screen,self.x,self.y)
        self.drawrect(screen)
         

class ballenv():

    def __init__(self,ballnum = 200):
        update_frequency = 5

        pygame.init()
        self.state_size = EYESHAPE
        self.action_size = 9
        #size = 1200,700
        self.scrRect = Rect(0,0,1200,700)
        self.screen = pygame.display.set_mode(self.scrRect.size,DOUBLEBUF)
        self.screen.set_alpha(None)
        self.ball = ball
        self.rlball = rlball
        self.ball.screen = self.scrRect
        self.rlball.screen = self.scrRect

        #number of init ball
        self.ballnum = int(ballnum)
        balls = []
        self.group = pygame.sprite.RenderUpdates()
        #player group
        self.player = pygame.sprite.RenderUpdates()
        #enemy group
        self.ball.containers = self.group
        self.rlball.containers = self.player
        self.rlball.enemygroup = self.group
        # build rlball
        self.playerball = rlball()
        #ball.collide = collide
        for i in range(self.ballnum):
            self.ball(color=array([200,200,200]))

        self.clock = pygame.time.Clock()

        #tf.reset_default_graph() #Clear the Tensorflow graph.
        #myAgent = agent(lr=1e-2,s_size=121,a_size=9,h_size=32) #Load the agent.
        #init = tf.global_variables_initializer()
     
    def reset(self):
        return np.zeros(EYESHAPE)
    
    def step(self, action):
        self.playerball.action = action
        # limit FPS to 60/s
        self.clock.tick(120)

        # pygame drawing
        self.screen.fill((0,0,0))
        self.player.update(self.screen)
        self.group.update(self.screen)
        
        # get reward state etc
        s1 = self.playerball.eyesView
        r = self.playerball.reward
        d = self.playerball.collided
        # print(s1)
        pygame.draw.rect(self.screen,(0,0,255),(GODWALL,
                                    GODWALL,
                                    self.scrRect.width-GODWALL*2,
                                    self.scrRect.height-GODWALL*2
                                ),
                        2)
        pygame.display.update();
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        
        return([s1,r,d])
        
         
if __name__ == "__main__":
        # test code
        env = ballenv()
        action = 4
        while 1:
            action = np.random.randint(0,9)
            s1,_,_ = env.step(action)
            #print(s1)
    
