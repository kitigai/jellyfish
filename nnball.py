import sys,pygame
from pygame.locals import *
from numpy.random import *
from numpy import *
from math import * 

GODPOWER = 0.5
GODWALL = 50
OMOMI_KEISUU = 10
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
        collided = pygame.sprite.spritecollide(self,group,False)
        if collided:
            for collideball in collided:
                if(collideball.size/OMOMI_KEISUU+fabs(collideball.sx)+fabs(collideball.sy) < self.size/OMOMI_KEISUU+fabs(self.sx)+fabs(self.sy)):
                   group.remove(collideball) 
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
            group.remove(self)
            for i in range(self.cal):
                x = normal(self.rect.centerx,50)
                y = normal(self.rect.centery,50)
                ball(x,y,self.power,self.span,self.color) 

    def drawcircle(self,screen):
        pygame.draw.circle(screen,self.color,(self.xint,self.yint),self.size)

    def drawrect(self,screen):
        #pygame.draw.rect(screen,self.color,(self.xint,self.yint)+self.rectsize)
        screen.fill(self.color,self.rect)



if __name__ == "__main__":
    
    pygame.init()
    #size = 1200,700
    scrRect = Rect(0,0,1200,700)
    screen = pygame.display.set_mode(scrRect.size)
    ball.screen = scrRect
    
    #number of init ball
    ballnum = int(sys.argv[1])
    balls = []
    group = pygame.sprite.RenderUpdates()

    collide = pygame.sprite.Group()
    ball.containers = group,collide
    ball.collide = collide
    for i in range(ballnum):
        ball()
   
    clock = pygame.time.Clock() 
    while 1:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x,y = event.pos
                ball(x,y)

        screen.fill((0,0,0))
        group.update(screen)
        pygame.draw.rect(screen,(0,0,255),(GODWALL,
                                    GODWALL,
                                    scrRect.width-GODWALL*2,
                                    scrRect.height-GODWALL*2
                                ),
                        2)
        pygame.display.update();
        
         
