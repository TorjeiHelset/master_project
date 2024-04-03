from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image

texture = None
w1 = 0
h1 = 0

def reshape(w, h):
    global w1
    global h1
    w1 = w
    h1 = h
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-w1/2, w1/2, -h1/2, h1/2)
    glMatrixMode(GL_MODELVIEW)

def background():
    glBindTexture(GL_TEXTURE_2D, texture)
    iw, ih = 800, 600
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(-iw/2, -ih/2)
    glTexCoord2f(1, 0)
    glVertex2f(iw/2, -ih/2)
    glTexCoord2f(1, 1)
    glVertex2f(iw/2, ih/2)
    glTexCoord2f(0, 1)
    glVertex2f(-iw/2, ih/2)
    glEnd()

def draw_on_top(x, y):
    glColor3f(0.3, 0.3, 0.3)
    glBegin(GL_LINES)
    glVertex2f(x, y)            # Arrow base
    glVertex2f(x + 0.1, y + 0.2) # Arrow tip
    glVertex2f(x, y)            # Arrow base
    glVertex2f(x - 0.1, y + 0.2) # Arrow tip
    glEnd()

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glEnable(GL_TEXTURE_2D)
    
    background()
    glDisable(GL_TEXTURE_2D)
    x_1, y_1 = [-0.5, 0.5]
    draw_on_top(x_1, y_1)
    
    glutSwapBuffers()

def load_texture(filename):
    img = Image.open(filename)
    img_data = img.tobytes("raw", "RGB", 0, -1)
    width, height = img.size

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    return texture

def main():
    global texture
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"OpenGL Window")

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    texture = load_texture("background.jpg")  # Replace "background.jpg" with your image file path
    glClearColor(0.0, 0.0, 0.0, 1.0)

    glutMainLoop()

if __name__ == "__main__":
    main()
