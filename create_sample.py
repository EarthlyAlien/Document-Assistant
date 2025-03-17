from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_sample_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    
    # First page
    c.setFont("Helvetica", 12)
    c.drawString(72, 750, "Introduction to Artificial Intelligence")
    c.drawString(72, 720, "Artificial Intelligence (AI) is a branch of computer science that aims to create")
    c.drawString(72, 700, "intelligent machines that can perform tasks that typically require human intelligence.")
    c.drawString(72, 680, "These tasks include:")
    c.drawString(92, 660, "- Visual perception")
    c.drawString(92, 640, "- Speech recognition")
    c.drawString(92, 620, "- Decision-making")
    c.drawString(92, 600, "- Language translation")
    
    # Second page
    c.showPage()
    c.setFont("Helvetica", 12)
    c.drawString(72, 750, "Machine Learning and Deep Learning")
    c.drawString(72, 720, "Machine Learning is a subset of AI that focuses on developing systems that can")
    c.drawString(72, 700, "learn and improve from experience without being explicitly programmed.")
    c.drawString(72, 680, "Deep Learning, a subset of Machine Learning, uses artificial neural networks")
    c.drawString(72, 660, "to learn from large amounts of data.")
    
    # Third page
    c.showPage()
    c.setFont("Helvetica", 12)
    c.drawString(72, 750, "Applications of AI")
    c.drawString(72, 720, "AI has numerous real-world applications, including:")
    c.drawString(92, 700, "1. Healthcare: Disease diagnosis and drug discovery")
    c.drawString(92, 680, "2. Finance: Fraud detection and algorithmic trading")
    c.drawString(92, 660, "3. Transportation: Self-driving vehicles and traffic prediction")
    c.drawString(92, 640, "4. Entertainment: Personalized content recommendations")
    
    c.save()

if __name__ == "__main__":
    create_sample_pdf("sample_ai.pdf") 