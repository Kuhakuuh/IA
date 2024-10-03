import { Routes } from '@angular/router';
import { TrainingComponent } from './components/training/training.component';
import { AppComponent } from './app.component';
import { MainComponent } from './components/main/main.component';

export const routes: Routes = [
    {path: '', component:MainComponent },
    {path: 'training', component:TrainingComponent}
];