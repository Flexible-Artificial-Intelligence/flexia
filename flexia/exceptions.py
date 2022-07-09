class LibraryException(Exception):
   def __init__(self,  library):
        self.message = f"Module `{library}` not found or not provided."
        super().__init__(self.message)