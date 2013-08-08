package org.lemming.outputs;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.interfaces.Output;

public class PrintToFile implements Output {
	Store<Localization> s;
	
	@Override
	public void setInput(Store<Localization> s) {
		this.s = s;
	}

	File f;
	FileWriter w;
	
	public PrintToFile(File f) {
		this.f = f;
	}
	
	@Override
	public void run() {
		try {
			w = new FileWriter(f);
			
			while(true) {
				Localization l = s.get();
				
				if (l != null)
					w.write(String.format("%d, %f, %f\n",l.getID(),l.getX(),l.getY()));
				
				w.flush();
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	protected void finalize() throws Throwable {
		if (w!=null) {
			w.close();
		}
	}
}
