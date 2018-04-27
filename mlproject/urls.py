from django.conf.urls import url
from django.contrib import admin
import views
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^send$',views.getMovies,name = "movies"),
    url(r'^movie$',views.getMovie,name = "movie"),
]
