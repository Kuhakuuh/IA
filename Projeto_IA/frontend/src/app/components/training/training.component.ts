import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatTableModule } from '@angular/material/table';

@Component({
  selector: 'app-training',
  standalone: true,
  imports: [CommonModule,MatTableModule,FormsModule],
  templateUrl: './training.component.html',
  styleUrl: './training.component.css'
})
export class TrainingComponent {


  activeTab: number = 1;


  setActiveTab(tab: number) {
    this.activeTab = tab;
  }

}
