package org.lemming.outputs;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import org.lemming.data.Localization;
import org.lemming.utils.LemMING;

public class PrintToFile extends SI<Localization> {
	
	File f;
	FileWriter w;

	public PrintToFile(File f) {
		this.f = f;
	}
	
	@Override
	public void beforeRun() {
		try {
			w = new FileWriter(f);
		} catch (IOException e) {
			e.printStackTrace();
			LemMING.error(e.getMessage());
		}
	}
	
	@Override
	public void afterRun() {
		try {
			w.close();
		} catch (IOException e) {
			e.printStackTrace();
			LemMING.error(e.getMessage());
		}
	}
	
	@Override
	public void process(Localization l) {
		try {
			w.write(String.format("%d, %f, %f\n",l.getID(),l.getX(),l.getY()));
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
}
