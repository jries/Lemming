package org.lemming.outputs;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;

import org.lemming.interfaces.Localization;
import org.lemming.utils.LemMING;

/**
 * @author Ronny Sczech
 *
 */
public class PrintToFile extends SingleInput<Localization> {
	
	private File f;
	private FileWriter w;
	private Locale curLocale;

	/**
	 * @param f - File
	 */
	public PrintToFile(File f) {
		this.f = f;
		this.curLocale = Locale.getDefault();
	}
	
	@Override
	public void beforeRun() {
		
		final Locale usLocale = new Locale( "en", "US" ); // setting us locale
		Locale.setDefault( usLocale );
		
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
		Locale.setDefault( curLocale );
	}
	
	@Override
	public void process(Localization l) {
		if (l==null) return;
		if(l.isLast()){ 
			stop();
			System.out.println("Export finished:"+l.getID()); 
			return;
		}
		try {
			String out = String.format("%d, %f, %f\n",l.getID(),l.getX(),l.getY());
			w.write(out);
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}

}
