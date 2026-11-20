import flask, cv2
print(f"Flask {flask.__version__} OK")
print(f"cv2 {cv2.__version__} OK")
try:
    import ultralytics
    print(f"ultralytics {ultralytics.__version__} OK")
except:
    print("ultralytics: tidak terinstall")
print("Python interpreter ini COCOK untuk Flask!")
