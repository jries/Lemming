package org.lemming.outputs;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;

import javolution.util.FastMap;

import org.lemming.data.LastElement;
import org.lemming.utils.LemMING;

/**
 * @author Ronny Sczech
 *
 */
public class GenericPrintToFile extends SingleInput<FastMap<String,Object>> {
	
	private Locale curLocale;
	private File f;
	private FileWriter w;

	/**
	 * @param f - file
	 */
	public GenericPrintToFile(File f) {
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
	public void process(FastMap<String,Object> elements) {
		if (elements==null) return;
		if (elements.isEmpty()) return;
		StringBuilder out = new StringBuilder();
		for (String key : elements.keySet()){
			Object o = elements.get(key);
			if(o instanceof LastElement){ 
				stop();
				System.out.println("Export finished"); 
				return;
			}
			out.append(o.toString());out.append(", ");
		}
		out.append("\n");
		try {
				
				w.write(out.toString());
			} catch (IOException e) {
				e.printStackTrace();
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

}
