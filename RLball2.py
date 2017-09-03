import sys,pygame
from pygame.locals import *
from numpy.random import *
from numpy import *
import numpy as np
from math import * 
import pdb
import tensorflow as tf
import tensorflow.contrib.slim as slim

try:
    xrange = xrange
except:
    xrange = range

# discount rate
gamma = 0.99

GODPOWER = 0.5
GODWALL = 50
OMOMI_KEISUU = 10

EYESHAPE=[11,11]
class distance:
    x = 100
    y = 100
    xint = 0
    yint = 0
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add =  0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

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
        collided = pygame.sprite.spritecollide(self ,group,False)
        if collided:
            for collideball in collided:
                if((collideball.size/OMOMI_KEISUU+fabs(collideball.sx)+fabs(collideball.sy) < self.size/OMOMI_KEISUU+fabs(self.sx)+fabs(self.sy))) and collideball.protected == False:
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
        if(self.count >= self.span):
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
        collided = pygame.sprite.spritecollide(self,group,True)
        if collided:
            self.collided = True
            for collideball in collided:
                self.reward = self.reward + collideball.cal
        
        # collision detection Eye vs other
        collideEye = pygame.sprite.groupcollide(self.eyesGroup,group,False,False)
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
         
class  agent():
    def __init__(self, lr, s_size, a_size, h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

if __name__ == "__main__":
    


    update_frequency = 5

    pygame.init()
    #size = 1200,700
    scrRect = Rect(0,0,1200,700)
    screen = pygame.display.set_mode(scrRect.size)
    ball.screen = scrRect
    rlball.screen = scrRect
    
    #number of init ball
    ballnum = int(sys.argv[1])
    balls = []
    group = pygame.sprite.RenderUpdates()
    #player group
    player = pygame.sprite.RenderUpdates()

    #collide = pygame.sprite.Group()
    #ball.containers = group,collide
    ball.containers = group
    rlball.containers = player
    #pdb.set_trace()
    playerball = rlball()
    #ball.collide = collide
    for i in range(ballnum):
        ball(color=array([200,200,200]))
   
    clock = pygame.time.Clock()
    
    tf.reset_default_graph() #Clear the Tensorflow graph.
    myAgent = agent(lr=1e-2,s_size=121,a_size=9,h_size=32) #Load the agent.
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        total_reward = []
        total_lenght = []
        
        gradBuffer = sess.run(tf.trainable_variables())
        running_reward = 0
        ep_history = []
        s = zeros(EYESHAPE)
        i = 0
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
        
        while 1:
            # choose action
            s = s.flatten()
            updatable = playerball.isupdatable()
            if(updatable):
                a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
                a = np.random.choice(a_dist[0],p=a_dist[0])
                # if there are no input agent always choose 0
                # to prevent that limit the provability
                if a < 0.2:
                    a = np.random.randint(0,9)
                    #print("random")
                else:
                    a = np.argmax(a_dist == a)
                    print(a)
                playerball.action = a
            clock.tick(60)

            # pygame event driven
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    x,y = event.pos
                    ball(x,y)

            # RLearning execute

            # pygame drawing
            screen.fill((0,0,0))
            player.update(screen)
            group.update(screen)
            
            # get reward state etc
            s1 = playerball.eyesView
            r = playerball.reward
            d = playerball.collided
            if(updatable):
                ep_history.append([s,a,r,s1])
                # print(s1)
                s = s1
            if d == True:
                print(i)
                # revive ball
                for xi in range(r):
                    ball(color=array([200,200,200]))
                # if collided then update
                # update network
                ep_history2 = np.array(ep_history)
                print(ep_history2.shape)
                ep_history2[:,2] = discount_rewards(ep_history2[:,2])
                feed_dict={myAgent.reward_holder:ep_history2[:,2],
                        myAgent.action_holder:ep_history2[:,1],myAgent.state_in:np.vstack(ep_history2[:,0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                #if i % update_frequency == 0 and i != 0:
                feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                _  = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                ep_history = []
                i = 0
                #total_reward.append(running_reward)
                #total_lenght.append(j)
            i = i + 1
            pygame.draw.rect(screen,(0,0,255),(GODWALL,
                                        GODWALL,
                                        scrRect.width-GODWALL*2,
                                        scrRect.height-GODWALL*2
                                    ),
                            2)
            pygame.display.update();
    # export graph
    #meta_graph_def = tf.train.export_meta_graph(filename='/tmp/my-model.meta')
        
         
