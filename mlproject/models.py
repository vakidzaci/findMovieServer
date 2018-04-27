from django.db import models



class Test(models.Model):
    # ...
    def __str__(self):
        return "Madik"
