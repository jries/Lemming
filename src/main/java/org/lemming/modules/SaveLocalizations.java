package org.lemming.modules;

import ij.IJ;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;

public class SaveLocalizations extends SingleRunModule {

	final private Locale curLocale;
	private File file;
	private FileWriter w;
	private int counter=0;

	public SaveLocalizations(File file) {
		this.curLocale = Locale.getDefault();
		this.file = file;
	}

	@Override
	public void beforeRun() {
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);

		try {
			w = new FileWriter(file);
		} catch (IOException e) {
			e.printStackTrace();
		}
		start = System.currentTimeMillis();
	}

	@Override
	public Element processData(Element data) {
		
		if (data.isLast()) {
			if (data instanceof Localization) {
				if (inputs.get(iterator).isEmpty()) {
					cancel();
					return null;
				}
				try {
					inputs.get(iterator).put(data);
				} catch (InterruptedException e) {}
				return null;
			}
			cancel();
			return null;
		}
		
		if (data instanceof Localization) {
			Localization loc = (Localization) data;
			
			try {
				w.write(loc.toString()+"⁄n");
			} catch (IOException e) {
				IJ.error("SaveLocalization:"+e.getMessage());;
			}
			counter++;
		
		}
		return null;
	}

	@Override
	public void afterRun() {
		try {
			w.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		Locale.setDefault(curLocale);
		System.out.println("" + counter + " localizations saved in " + (System.currentTimeMillis()-start) + "ms.");
	}

	@Override
	public boolean check() {
		return inputs.size()==1;
	}

}
