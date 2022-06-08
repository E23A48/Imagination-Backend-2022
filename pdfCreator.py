from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Frame
from reportlab.platypus import Image
from pathlib import Path
from reportlab.platypus import Spacer
import tempfile

def addFrame(img, string, x, y, width, height, canvas):
  story = []

  imagen_logo = Image(img, width=120, height=120)
  story.append(imagen_logo)

  story.append(Spacer(0,20))

  story.append(Paragraph(string))

  frame = Frame(x, y, width, height, showBoundary=1)

  frame.addFromList(story, canvas)

def createPDF(imgs, strings):
  Path("pdfs").mkdir(parents=True, exist_ok=True)
  
  filename = 'pdfs/'+next(tempfile._get_candidate_names())+'.pdf'

  c = Canvas(filename, pagesize=landscape(A4))

  w, h = landscape(A4)

  plus = 0
  for img, string in zip(imgs, strings):
    addFrame(img, string, 60 + plus, h - 350, 120, 190, c)
    plus += 140

  c.save()

  return filename