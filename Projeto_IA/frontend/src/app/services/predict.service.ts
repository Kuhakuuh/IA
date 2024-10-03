import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { House } from '../models/House';
import { Observable, catchError, map } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PredictService {
  Url = "http://localhost:8000/prediction/";
  

  httpOptions = {
    headers: new HttpHeaders({ 'Content-Type': 'application/json' })
  }

  constructor(private http: HttpClient) { }

  getAllPredicted(): Observable<House[]> {
    return this.http.get<{ predictions: House[] }>(`${this.Url}list`)
      .pipe(
        map(response => response.predictions),
        catchError(this.handleError<House[]>())
      );
  }
  createPredict(house: House): Observable<House> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    return this.http.post<House>(`${this.Url}create`, house, { headers })
      .pipe(
        catchError(this.handleError<House>())
      );
  }

  deletePredict(id: string): Observable<any> {
    return this.http.delete<any>(`${this.Url}delete/${id}`)
      .pipe(
        catchError(this.handleError<any>())
      );
  }

  getModels(): Observable<any[]> {
    return this.http.get<{ models: any[] }>(`http://localhost:8000/models`)
      .pipe(
        map(response => response.models),
        catchError(this.handleError<any[]>())
      );
  }



  private handleError<T>() {
    return (error: any): Observable<T> => {
      console.error(error);
      throw error;
    }
  }

  

}