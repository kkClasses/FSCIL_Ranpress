def get_model(model_name, args):
    name = model_name.lower()
    if  name == 'ranpac':
        from models.ranPress import Learner
    else:
        assert 0
    
    return Learner(args)
