# **Project Spam Detection**

As a software developer, email is one of the very important tool for communication.
In today's world we receive lotsof spam and we would want to design a system that can detect messages and mark it as spam.

**Progress**
- Organized the project into multiple folder structure
    - configuration: All the config information for the project resides
    - data: this is where the raw data and the processed data would reside
    - Libraries: all class module for the project will reside
    - models: if we import the models as pickle file, this is where it would be kept
    - notebook: all jupyter notebook files would reside here.
    - references: any additional references would be hed here
    - src: all relevant source files should be kept here
- created a class file for spam detection
    - The class will automatically detect the best model based on the training data and the test data.
    - It will hold the model with the best accuracy wihch can be used by the calling function to predict

**TODO**:
- Build a Model that detect Spam and mark it as a spam [Done]
- Build a System that can update its database with more spam data and refresh the model as and when it learns something new. 
    - This needs to be done either as event based or periodically run the machine learning process.
    - Build the system, by using software architecture and design principles. The Software should be scalable and can be used by external entities.
- Train data should be stored in a secure way that no one can tamper with it easily.
- Build documentation for your software

