import { Component, OnInit } from '@angular/core';
import { PredictService } from '../../services/predict.service';
import { House } from '../../models/House';
import { CommonModule } from '@angular/common';
import { catchError } from 'rxjs';
import { MatTableDataSource, MatTableModule } from '@angular/material/table';
import { trigger } from '@angular/animations';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
@Component({
  selector: 'app-main',
  standalone: true,
  imports: [CommonModule,MatTableModule,FormsModule],
  templateUrl: './main.component.html',
  styleUrl: './main.component.css'
})
export class MainComponent implements OnInit {

  houses: House[] = [];
  models: any[] = [];
  selectedModel: string | undefined;
  house: House = {
    id:'',
    Bedrooms: '',
    Grade: '',
    Lat: '',
    Living_m2: '',
    Long: '',
    Lot_m2: '',
    Price_Predicted:0,
    bathrooms: '',
    Selected_model: '' 
  };
  isContentVisible: boolean | undefined;
  expandedIndex: number | null | undefined;

  plotUrl: SafeUrl | undefined;
  constructor(
    private predictServices : PredictService,
    private sanitizer: DomSanitizer
  ){}


  ngOnInit(): void{
    this.getHouses();
    this.getModels();
  
  }
  

  getHouses(): void {
    this.predictServices.getAllPredicted().subscribe(
      (data: House[]) => {
        this.houses = data;
        console.log(this.houses[0])
      },
      error => {
        console.error('There was an error!', error);
      }
    );
  }

  createHouse(): void {
    if ( this.house.Selected_model == '' ){
      alert("Please select the model!")
    }
    else{
      this.predictServices.createPredict(this.house).subscribe(
        (response) => {
         console.log("Added "+response);
          //this.resetNewHouse();
        },
        error => {
          console.error('There was an error!', error);
        }
      );
      
    }
   
    this.getHouses();
  }

  delete(id:string):void {
    this.predictServices.deletePredict(id).subscribe(
      (data)=> {
        console.log("Deleted "+data);
      },
      error => {
        console.error('There was an error!', error);
      }
         
    );
   this.getHouses();
  }

  getModels():void {
    this.predictServices.getModels().subscribe(
      (data)=>{
        console.log(data)
      this.models =data
      },
      error => {
        console.error('There was an error!', error);
      }
    );
  }

  expandCard(index: number): void {
    this.expandedIndex = this.expandedIndex === index ? null : index;
  }
  deletePrediction(id:string): void{
    this.delete(id)

    this.getHouses();
  }

  resetNewHouse(): void {
    this.house = {
      id:'',
      Bedrooms: '',
      Grade: '',
      Lat: '',
      Price_Predicted: 0,
      Living_m2: '',
      Long: '',
      Lot_m2: '',
      bathrooms: '',
    };
    this.getHouses();
  }
  updateValue(model: string) {
    this.selectedModel = model;
    this.house.Selected_model = model;
  }



  
  

}
